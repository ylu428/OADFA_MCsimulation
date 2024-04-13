# OADFA_MCsimulation
Monte Carlo Simulation of Sequential two-photon delayed fluorescence anisotropy



## Sequential Two-Photon Delayed Fluorescence Anisotropy Simulation
### Overview
This repository provides a Python-based simulation tool for calculating sequential two-photon delayed fluorescence anisotropy. The tool allows users to specify various physical properties of fluorophores and laser parameters to simulate the resulting fluorescence anisotropy decay using Monte Carlo simulation.

### Features
#### Fluorophore Properties: Users can define the following properties:

- Fluorescence lifetime
- Quantum yield
- Extinction coefficient
- Molecular size
- And more...
#### Laser Properties: Users can specify the following laser parameters:

- Wavelength
- Pulse width
- Intensity
- And more...

#### Monte Carlo Simulation: 
- The rotational Brownian motion and optically activated delayed fluorescence (OADF) of the fluorophore are calculated using Monte Carlo simulation based on the provided parameters.
- The fluorescence anisotropy decay is calculated by combining the result of rotational Brownian motion OADF.
- Fluorescence anisotropy is a bulk analysis technique, therefore users can specify the number of molecules participating in the excitation-emission cycle. Increasing the number of molecules enhances result accuracy by reducing noise, but it also extends simulation time.
- For simulating regular fluorescence anisotropy using a single excitation laser, set the intensity of the secondary laser to zero.
- A polarization analyzer can be incorporated into the system.
- The resulting anisotropy decay is fitted with an exponential decay function to calculate the rotational correlation time.
- Simulation results can be exported as CSV files.
- Imported CSV files can undergo further analysis.
