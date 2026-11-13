// CPU kernel for kc_string_forward.py. No autograd graph or reverse sweep.
// Every Dual is exactly one displacement and ten forward derivatives (eight
// material parameters, then the pluck and pickup positions).
#include <torch/extension.h>
#include <array>
#include <cstring>
#include <vector>

constexpr int PARAMETERS = 10;
constexpr int COLUMNS = PARAMETERS + 1;

struct Dual {
    std::array<double, COLUMNS> v{};
};
static_assert(sizeof(Dual) == COLUMNS * sizeof(double));

inline Dual operator+(const Dual& a, const Dual& b) {
    Dual result;
    for (int j = 0; j < COLUMNS; ++j) result.v[j] = a.v[j] + b.v[j];
    return result;
}
inline Dual operator-(const Dual& a, const Dual& b) {
    Dual result;
    for (int j = 0; j < COLUMNS; ++j) result.v[j] = a.v[j] - b.v[j];
    return result;
}
inline Dual operator*(const Dual& a, const Dual& b) {
    Dual result;
    result.v[0] = a.v[0] * b.v[0];
    // Product rule for each parameter, with the primal stored in column zero.
    for (int j = 1; j < COLUMNS; ++j)
        result.v[j] = a.v[j] * b.v[0] + a.v[0] * b.v[j];
    return result;
}
inline Dual operator*(double scale, const Dual& a) {
    Dual result;
    for (int j = 0; j < COLUMNS; ++j) result.v[j] = scale * a.v[j];
    return result;
}
inline Dual operator/(const Dual& a, const Dual& b) {
    Dual result;
    result.v[0] = a.v[0] / b.v[0];
    for (int j = 1; j < COLUMNS; ++j)
        result.v[j] = (a.v[j] - result.v[0] * b.v[j]) / b.v[0];
    return result;
}
inline Dual operator/(const Dual& a, double scale) {
    Dual result;
    for (int j = 0; j < COLUMNS; ++j) result.v[j] = a.v[j] / scale;
    return result;
}

// Order 2: the reference code's three-point stencil with zero end points.
// Order 4: the five-point stencil (-1, 16, -30, 16, -1)/12 closed by odd reflection
// about the fixed ends (u_{-1} = u_n = 0, u_{-1-m} = -u_{m-1}, u_{n+m} = -u_{n-m}),
// accumulated in ascending offset order exactly like kc_string_forward.second_derivative.
void second_derivative(const std::vector<Dual>& u, std::vector<Dual>& result,
                       const Dual& inverse_spacing_squared, int order) {
    const auto n = static_cast<std::ptrdiff_t>(u.size());
    if (order == 2) {
        result[0] = (-2.0 * u[0] + u[1]) * inverse_spacing_squared;
        for (std::ptrdiff_t i = 1; i + 1 < n; ++i)
            result[i] = (u[i - 1] - 2.0 * u[i] + u[i + 1]) * inverse_spacing_squared;
        result[n - 1] = (u[n - 2] - 2.0 * u[n - 1]) * inverse_spacing_squared;
        return;
    }
    auto at = [&](std::ptrdiff_t i) -> Dual {
        if (i >= 0 && i < n) return u[i];
        if (i == -1 || i == n) return Dual{};
        return -1.0 * (i < -1 ? u[-i - 2] : u[2 * n - i]);
    };
    for (std::ptrdiff_t i = 0; i < n; ++i)
        result[i] = (((((-1.0 / 12) * at(i - 2) + (4.0 / 3) * at(i - 1)) + (-2.5) * u[i])
                      + (4.0 / 3) * at(i + 1)) + (-1.0 / 12) * at(i + 2)) * inverse_spacing_squared;
}

// The pickup is a dual weight vector (value plus tangents, nonzero only in the
// pickup column): the sample is its product-rule dot with the state. On a node
// the value column is an exact one-hot, so adding the other terms is exact.
inline Dual readout(const std::vector<Dual>& weights, const std::vector<Dual>& state) {
    Dual sample;
    for (size_t i = 0; i < state.size(); ++i) sample = sample + weights[i] * state[i];
    return sample;
}

std::vector<torch::Tensor> run(torch::Tensor states, torch::Tensor coefficients, torch::Tensor weights,
                                        torch::Tensor loss, int64_t steps,
                                        double amplitude_squared, int64_t output_stride,
                                        bool include_initial, bool need_tangents, bool accumulate_loss,
                                        int64_t order) {
    TORCH_CHECK(order == 2 || order == 4, "Spatial order must be 2 or 4");
    for (const auto& tensor : {states, coefficients, weights, loss}) {
        TORCH_CHECK(tensor.device().is_cpu() && tensor.scalar_type() == torch::kFloat64,
                    "Expected CPU float64 tensors");
        TORCH_CHECK(tensor.is_contiguous(), "Expected contiguous dual columns");
        TORCH_CHECK(!tensor.requires_grad(), "Kernel uses explicit forward derivatives, not autograd");
    }
    TORCH_CHECK(states.dim() == 3 && states.size(0) == 2 && states.size(1) >= 3 && states.size(2) == COLUMNS,
                "Expected states with shape (2, interior_points, 11)");
    TORCH_CHECK(coefficients.dim() == 2 && coefficients.size(0) == 8 && coefficients.size(1) == COLUMNS,
                "Expected coefficients with shape (8, 11)");
    TORCH_CHECK(weights.dim() == 2 && weights.size(0) == states.size(1) && weights.size(1) == COLUMNS,
                "Expected pickup weights with shape (interior_points, 11)");
    TORCH_CHECK(loss.dim() == 1 && loss.size(0) == COLUMNS && steps >= 0 && amplitude_squared > 0 && output_stride > 0,
                "Invalid loss, step count, or amplitude normalizer");
    const auto n = static_cast<size_t>(states.size(1));
    std::vector<Dual> previous(n), current(n), following(n), second(n), fourth(n), previous_second(n);
    std::vector<Dual> direction(n), linear(n), mass_scaled_rhs(n), mass_scaled_direction(n), pickup(n);
    std::array<Dual, 8> c;
    Dual loss_sum;
    std::memcpy(previous.data(), states.data_ptr<double>(), n * sizeof(Dual));
    std::memcpy(current.data(), states.data_ptr<double>() + n * COLUMNS, n * sizeof(Dual));
    std::memcpy(pickup.data(), weights.data_ptr<double>(), n * sizeof(Dual));
    std::memcpy(c.data(), coefficients.data_ptr<double>(), sizeof(c));
    std::memcpy(&loss_sum, loss.data_ptr<double>(), sizeof(Dual));
    const int64_t total_samples = steps + (include_initial ? 2 : 0);
    const int64_t output_count = total_samples == 0 ? 0 : (total_samples - 1) / output_stride + 1;
    auto samples = torch::empty({output_count}, states.options());
    auto tangents = torch::empty({need_tangents ? output_count : 0, PARAMETERS}, states.options());
    auto samples_data = samples.data_ptr<double>();
    auto tangents_data = tangents.data_ptr<double>();
    auto record_sample = [&](const Dual& sample, int64_t index) {
        if (index % output_stride != 0) return;
        const auto row = index / output_stride;
        samples_data[row] = sample.v[0];
        if (need_tangents) std::memcpy(tangents_data + row * PARAMETERS, sample.v.data() + 1, PARAMETERS * sizeof(double));
    };
    if (include_initial) {
        record_sample(readout(pickup, previous), 0);
        record_sample(readout(pickup, current), 1);
    }
    for (int64_t step = 0; step < steps; ++step) {
        second_derivative(current, second, c[0], static_cast<int>(order));
        second_derivative(second, fourth, c[0], static_cast<int>(order));
        second_derivative(previous, previous_second, c[0], static_cast<int>(order));
        Dual projected_previous;
        for (size_t i = 0; i < n; ++i) {
            linear[i] = (2.0 * c[1]) * current[i] + c[2] * second[i] - c[3] * fourth[i]
                        + c[4] * previous[i] - c[5] * previous_second[i];
            direction[i] = c[7] * second[i];
            projected_previous = projected_previous + direction[i] * previous[i];
        }
        Dual denominator, projection;
        for (size_t i = 0; i < n; ++i) {
            const Dual rhs = linear[i] - direction[i] * projected_previous;
            mass_scaled_rhs[i] = c[6] * rhs;
            mass_scaled_direction[i] = c[6] * direction[i];
            denominator = denominator + direction[i] * mass_scaled_direction[i];
            projection = projection + direction[i] * mass_scaled_rhs[i];
        }
        denominator.v[0] += 1.0;
        const Dual correction = projection / denominator;
        for (size_t i = 0; i < n; ++i)
            following[i] = mass_scaled_rhs[i] - mass_scaled_direction[i] * correction;
        const Dual sample = readout(pickup, following);
        record_sample(sample, step + (include_initial ? 2 : 0));
        if (accumulate_loss) loss_sum = loss_sum + (sample * sample) / amplitude_squared;
        previous.swap(current);
        current.swap(following);
    }
    auto final_states = torch::empty_like(states);
    auto final_loss = torch::empty_like(loss);
    std::memcpy(final_states.data_ptr<double>(), previous.data(), n * sizeof(Dual));
    std::memcpy(final_states.data_ptr<double>() + n * COLUMNS, current.data(), n * sizeof(Dual));
    std::memcpy(final_loss.data_ptr<double>(), &loss_sum, sizeof(Dual));
    return {final_states, final_loss, samples, tangents};
}

std::vector<torch::Tensor> advance_block(torch::Tensor states, torch::Tensor coefficients, torch::Tensor weights,
                                        torch::Tensor loss, int64_t steps, double amplitude_squared,
                                        int64_t order) {
    auto result = run(states, coefficients, weights, loss, steps, amplitude_squared, 1, false, false, true, order);
    return {result[0], result[1], result[2]};
}

std::vector<torch::Tensor> simulate(torch::Tensor states, torch::Tensor coefficients, torch::Tensor weights,
                                   int64_t steps, int64_t output_stride, int64_t order) {
    auto loss = torch::zeros({COLUMNS}, states.options());
    auto result = run(states, coefficients, weights, loss, steps, 1.0, output_stride, true, true, false, order);
    return {result[2], result[3]};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("simulate", &simulate, pybind11::call_guard<pybind11::gil_scoped_release>());
    module.def("advance_block", &advance_block, pybind11::call_guard<pybind11::gil_scoped_release>());
}
