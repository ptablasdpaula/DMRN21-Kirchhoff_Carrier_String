"""Damped non-SAV Kirchhoff–Carrier string.

Derivations and reference conventions: ../notebooks/kc_string.ipynb.
Numerical expressions intentionally preserve KCStringSAV_noSplit.m.
"""

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


def squared_wavenumber(angular_frequency, wave_speed, stiffness):
    """Positive root of K² z² + c² z - omega² = 0, where z = wavenumber².

    See notebook §2: this uses the undamped, linear dispersion relation.
    """
    dispersion_discriminant = wave_speed**4 + 4 * stiffness**2 * angular_frequency**2
    return (-wave_speed**2 + math.sqrt(dispersion_discriminant)) / (2 * stiffness**2)


def damping_coefficients(wavenumber_squared_1, wavenumber_squared_2, t60_1, t60_2):
    """Fit sigma_0 + sigma_1*z through the reference's two decay targets.

    Returns sigma_0 [1/s], sigma_1 [m²/s]. The factor 6 matches MATLAB;
    standard amplitude T60 with this damping operator uses 3 (notebook §3).
    """
    reference_decay_rate_1 = 6 * math.log(10) / t60_1
    reference_decay_rate_2 = 6 * math.log(10) / t60_2
    wavenumber_squared_difference = wavenumber_squared_2 - wavenumber_squared_1
    sigma_1 = (reference_decay_rate_2 - reference_decay_rate_1) / wavenumber_squared_difference
    sigma_0 = reference_decay_rate_1 - sigma_1 * wavenumber_squared_1
    return sigma_0, sigma_1


wavenumber_squared_ref_1 = squared_wavenumber(omega_ref_1_rad_s, wave_speed, stiffness)
wavenumber_squared_ref_2 = squared_wavenumber(omega_ref_2_rad_s, wave_speed, stiffness)
sigma_0, sigma_1 = damping_coefficients(
    wavenumber_squared_ref_1, wavenumber_squared_ref_2,
    t60_ref_1_seconds, t60_ref_2_seconds,
)

if not damp_on:
    sigma_0 = 0.0
    sigma_1 = 0.0

sigma_L = sigma_0  # Alias used in the MATLAB reference implementation.


def minimum_grid_spacing(time_step, wave_speed, stiffness, sigma_1):
    """Positive h threshold from h⁴ - (c² dt² + 4 sigma_1 dt) h² - 4 K² dt² = 0.

    Notebook §4 derives this bound from the linear damped difference scheme.
    The safety margin is applied separately by simulation_grid().
    """
    wave_and_damping_term = wave_speed**2 * time_step**2 + 4 * sigma_1 * time_step
    spacing_discriminant = wave_and_damping_term**2 + 16 * stiffness**2 * time_step**2
    minimum_spacing_squared = (wave_and_damping_term + math.sqrt(spacing_discriminant)) / 2
    return math.sqrt(minimum_spacing_squared)


def stable_time_step(grid_spacing, wave_speed, stiffness, sigma_1, stability_margin=0.01):
    """Invert h = (1 + margin) * h_min(dt), taking the positive dt root.

    Notebook §4 gives the quadratic in dt. This factored expression keeps
    the reference's arithmetic order; grid_spacing is in m and the result in s.
    """
    margin_factor = 1 + stability_margin
    time_step_radicand = (
        4 * (stiffness**2 + sigma_1**2) * margin_factor**2
        + wave_speed**2 * grid_spacing**2
    )
    time_step_denominator = margin_factor * (
        4 * stiffness**2 * margin_factor**2 + wave_speed**2 * grid_spacing**2
    )
    return grid_spacing**2 * (
        math.sqrt(time_step_radicand) - 2 * sigma_1 * margin_factor
    ) / time_step_denominator


def simulation_grid(oversampling_factor=oversampling_factor):
    """Return (interval_count, spacing_m, time_step_s, reported_sample_rate_hz).

    The even interval count places a node at L/2. As in MATLAB, the reported
    integer sample rate is floor(1/dt); it does not replace dt (notebook §4).
    """
    stability_margin = 0.01
    requested_time_step = 1 / (oversampling_factor * base_sample_rate)
    initial_grid_spacing = minimum_grid_spacing(
        requested_time_step, wave_speed, stiffness, sigma_1
    ) * (1 + stability_margin)
    grid_interval_count = 2 * math.floor(0.5 * length / initial_grid_spacing)
    if grid_interval_count < 4:
        raise ValueError("The selected grid has too few points for the string update.")
    grid_spacing = length / grid_interval_count
    time_step = stable_time_step(
        grid_spacing, wave_speed, stiffness, sigma_1, stability_margin
    )
    sample_rate = math.floor(1 / time_step)
    return grid_interval_count, grid_spacing, time_step, sample_rate


def solve_rank_one(right_hand_side, nonlinear_direction, inverse_mass):
    """Solve (m Id + q qᵀ) u = b by Sherman–Morrison; see notebook §6.

    inverse_mass is 1 / [rho*A*(1 + sigma_0*dt)], not a matrix.
    """
    mass_scaled_rhs = inverse_mass * right_hand_side
    mass_scaled_direction = inverse_mass * nonlinear_direction
    correction_denominator = 1 + torch.dot(nonlinear_direction, mass_scaled_direction)
    projected_rhs = torch.dot(nonlinear_direction, mass_scaled_rhs)
    return mass_scaled_rhs - mass_scaled_direction * (projected_rhs / correction_denominator)


def second_spatial_derivative(values, grid_spacing):
    """Apply D² with zero endpoint values (notebook §5; paper §6).

    Applying this twice also sets endpoint curvature to zero: simply supported
    bending conditions, not clamped ends.
    """
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

    grid_interval_count, grid_spacing, time_step, sample_rate = simulation_grid(
        oversampling_factor
    )
    time_samples = math.floor(sample_rate * duration_seconds)
    spatial_points = grid_interval_count - 1

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

    # Reference initType = 1. Extra h factors are preserved, not a standard
    # Taylor step of the pointwise PDE; see notebook §5 for the comparison.
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

    # Notebook §6: (m Id + q qᵀ) u_next = linear_term - q (qᵀ u_previous).
    # This is the non-SAV benchmark, related to paper Eq. (73), with damping.
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
            previous_projection = torch.dot(q, previous_displacement)
            right_hand_side = linear_term - q * previous_projection
            next_displacement = solve_rank_one(right_hand_side, q, inverse_mass)

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
