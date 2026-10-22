import math
import torch


# Simulation settings from KCStringSAV_noSplit.m
SAV = False
damp_on = True

base_sample_rate = 44_100
oversampling_factor = 512
duration_seconds = 0.06
amplitude = 2e-3

# String parameters
material_density = 8.05 * 10**3  # rho, kg/m^3
tension = 75  # T0, N
radius = 3.5560e-4  # r, m (0.016 gauge)
young_modulus = 174e9  # E, Pa
length = 1  # L, m

area = math.pi * radius**2  # A, m^2
linear_mass_density = material_density * area  # rho*A, kg/m
moment_of_inertia = math.pi * radius**4 / 4  # I, m^4
stiffness = math.sqrt(
    young_modulus * moment_of_inertia / linear_mass_density
)  # K, m^2/s
wave_speed = math.sqrt(tension / linear_mass_density)  # c, m/s

# Damping reference points: each T60 is specified at its corresponding frequency.
damping_ref_freq_1_hz = 0.0
damping_ref_freq_2_hz = 1000.0
t60_ref_1_seconds = 15.0
t60_ref_2_seconds = 10.0

omega_ref_1_rad_s = 2 * math.pi * damping_ref_freq_1_hz
omega_ref_2_rad_s = 2 * math.pi * damping_ref_freq_2_hz

# Invert omega^2 = c^2*k^2 + K^2*k^4 to get k^2 at each reference frequency.
wavenumber_squared_ref_1 = (
    -wave_speed**2
    + math.sqrt(wave_speed**4 + 4 * stiffness**2 * omega_ref_1_rad_s**2)
) / (2 * stiffness**2)
wavenumber_squared_ref_2 = (
    -wave_speed**2
    + math.sqrt(wave_speed**4 + 4 * stiffness**2 * omega_ref_2_rad_s**2)
) / (2 * stiffness**2)

# Convert the target T60 values to decay rates, then fit rate = sigma_0 + sigma_1*k^2.
target_decay_rate_ref_1 = 6 * math.log(10) / t60_ref_1_seconds
target_decay_rate_ref_2 = 6 * math.log(10) / t60_ref_2_seconds
delta_wavenumber_squared = wavenumber_squared_ref_2 - wavenumber_squared_ref_1
sigma_1 = (target_decay_rate_ref_2 - target_decay_rate_ref_1) / delta_wavenumber_squared
sigma_0 = target_decay_rate_ref_1 - sigma_1 * wavenumber_squared_ref_1

if not damp_on:
    sigma_0 = 0.0
    sigma_1 = 0.0

sigma_L = sigma_0  # Alias used in the MATLAB reference implementation.


def second_spatial_derivative(values, grid_spacing):
    """Apply the reference's centered second difference with fixed-end boundaries."""
    derivative = torch.empty_like(values)
    scale = 1 / grid_spacing**2

    derivative[0] = (-2 * values[0] + values[1]) * scale
    derivative[-1] = (values[-2] - 2 * values[-1]) * scale
    derivative[1:-1] = (
        values[:-2] - 2 * values[1:-1] + values[2:]
    ) * scale

    return derivative


def simulate_string(
    duration_seconds=duration_seconds,
    oversampling_factor=oversampling_factor,
):
    """Return the midpoint displacement samples and their sample rate.

    This follows the explicit, non-SAV update in KCStringSAV_noSplit.m with
    damping enabled. The returned samples are displacements in metres.
    """
    if SAV:
        raise ValueError("This implementation follows the SAV = false branch only.")
    if not damp_on:
        raise ValueError("This configuration is intended to run with damp_on = True.")

    stability_margin = 0.01
    time_step = 1 / (oversampling_factor * base_sample_rate)

    damping_time_term = wave_speed**2 * time_step**2 + 4 * sigma_1 * time_step
    initial_grid_spacing = math.sqrt(
        (
            damping_time_term
            + math.sqrt(damping_time_term**2 + 16 * stiffness**2 * time_step**2)
        )
        / 2
    ) * (1 + stability_margin)

    # The MATLAB code uses an even number of intervals so the output is at the centre.
    grid_interval_count = 2 * math.floor(0.5 * length / initial_grid_spacing)
    grid_spacing = length / grid_interval_count

    # Recalculate the stable time step for the selected spatial grid.
    stability_radicand = (
        4 * (stiffness**2 + sigma_1**2) * (1 + stability_margin) ** 2
        + wave_speed**2 * grid_spacing**2
    )
    stability_denominator = (1 + stability_margin) * (
        4 * stiffness**2 * (1 + stability_margin) ** 2
        + wave_speed**2 * grid_spacing**2
    )
    time_step = grid_spacing**2 * (
        math.sqrt(stability_radicand) - 2 * sigma_1 * (1 + stability_margin)
    ) / stability_denominator

    sample_rate = math.floor(1 / time_step)
    time_samples = math.floor(sample_rate * duration_seconds)
    spatial_points = grid_interval_count - 1

    if spatial_points < 2:
        raise ValueError("The selected grid has too few points for the string update.")

    # Spatial coordinates are the interior points; both string ends are fixed at zero.
    point_indices = torch.arange(1, grid_interval_count, dtype=torch.float64)
    previous_displacement = amplitude * torch.sin(
        math.pi * point_indices / grid_interval_count
    )

    previous_second_derivative = second_spatial_derivative(
        previous_displacement, grid_spacing
    )
    previous_fourth_derivative = second_spatial_derivative(
        previous_second_derivative, grid_spacing
    )
    nonlinear_initial_term = (
        -torch.dot(previous_displacement, previous_second_derivative)
        * previous_second_derivative
    )

    # This is the reference script's first-step initialization for initType = 1.
    displacement = previous_displacement + 0.5 * time_step**2 * (
        grid_spacing * wave_speed**2 * previous_second_derivative
        - grid_spacing * stiffness**2 * previous_fourth_derivative
        + grid_spacing
        * (0.5 * young_modulus * grid_spacing / (length * material_density))
        * nonlinear_initial_term
    )

    output_position = (spatial_points + 1) // 2 - 1  # MATLAB's one-based midpoint index.
    output_sound = torch.empty(time_samples + 2, dtype=torch.float64)
    output_sound[0] = previous_displacement[output_position]
    output_sound[1] = displacement[output_position]

    # The rank-one update below is the explicit non-SAV scheme in the MATLAB file.
    inverse_mass = 1 / (linear_mass_density + linear_mass_density * sigma_0 * time_step)
    q_scale = time_step * math.sqrt(
        grid_spacing * young_modulus * area / length
    ) * 0.5

    with torch.no_grad():
        for step in range(time_samples):
            displacement_second_derivative = second_spatial_derivative(
                displacement, grid_spacing
            )
            displacement_fourth_derivative = second_spatial_derivative(
                displacement_second_derivative, grid_spacing
            )
            previous_second_derivative = second_spatial_derivative(
                previous_displacement, grid_spacing
            )

            linear_term = (
                2 * linear_mass_density * displacement
                + (tension * time_step**2 + 2 * linear_mass_density * sigma_1 * time_step)
                * displacement_second_derivative
                - young_modulus
                * moment_of_inertia
                * time_step**2
                * displacement_fourth_derivative
                + (linear_mass_density * sigma_0 * time_step - linear_mass_density)
                * previous_displacement
                - 2
                * linear_mass_density
                * sigma_1
                * time_step
                * previous_second_derivative
            )

            q = q_scale * displacement_second_derivative
            coefficient_0 = torch.dot(q, previous_displacement)
            coefficient_1 = linear_term - q * coefficient_0
            coefficient_2 = inverse_mass * coefficient_1
            coefficient_3 = inverse_mass * q
            coefficient_4 = 1 + torch.dot(q, coefficient_3)
            coefficient_5 = torch.dot(q, coefficient_2)
            next_displacement = coefficient_2 - coefficient_3 * (
                coefficient_5 / coefficient_4
            )

            output_sound[step + 2] = next_displacement[output_position]
            previous_displacement = displacement
            displacement = next_displacement

    return output_sound, sample_rate


if __name__ == "__main__":
    output_sound, output_sample_rate = simulate_string()
    print(
        f"Generated {output_sound.numel():,} midpoint samples "
        f"at {output_sample_rate:,} Hz."
    )
