# Technical Documentation: Exoplanet Search (Physical Principles)

## Introduction

Exoplanet detection relies on two key observational methods: the transit method and the radial velocity (RV) method. Together, they allow determination of both the planet's size and mass, and consequently its density, helping to classify the object (gas giant, rocky planet, super-Earth, etc.).

<video width="640" height="360" controls>
  <source src="images/transit_method_multiple_planets_4K.mp4" type="video/mp4">
</video>

---

## 1. Transit Method

### Principle

When a planet passes between its star and the observer, it partially blocks the star's disk. This causes a drop in observed stellar brightness, known as a transit.

### Key Observables

* **Transit depth (δ):** defined as the ratio of areas:

$$
\delta = \left( \frac{R_p}{R_*} \right)^2
$$

  From this, the planet radius can be determined if the star's radius is known.
* **Periodic dips:** repeated, identical dips in the light curve reveal the orbital period.

### What can be measured from transit

* Planet radius $R_p$.

* Orbital semi-major axis $a$ via Kepler's third law:

$$ a^3 = \frac{G M_* P^2}{4\pi^2}$$

* Planet equilibrium temperature:

$$ T_{eq} = T_* \left( \frac{R_*}{2a} \right)^{1/2} (1 - A)^{1/4}, \ \ \text{where } A \text{ is albedo}.$$

### Limitations of the transit method

* Only radius can be measured, not mass.  
* Transits occur only for favorable orbital orientations ($i \approx 90^\circ$).  
* False positives possible (eclipsing binaries, background sources).

---

## 2. Radial Velocity Method (Doppler Effect)

### Principle

The planet and star orbit their common center of mass. The star’s orbital motion produces shifts in its spectral lines (Doppler effect).  

* Moving toward the observer → lines shift blue.  
* Moving away → lines shift red.  

## Radial Velocity Visualization

<video width="640" height="360" controls>
  <source src="images/radial_velocity-1.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Key Formulas

* **Doppler shift:**

$$ 
\frac{\Delta \lambda}{\lambda} = \frac{v_{rad}}{c}, 
$$ 

where $v_{rad}$ is the star’s radial velocity.

* **Signal semi-amplitude (K):**

$$ K \approx \left( \frac{2\pi G}{P} \right)^{1/3} \frac{M_p \sin i}{M_*^{2/3}} \frac{1}{\sqrt{1 - e^2}}, $$

where $M_p$ = planet mass, $M_*$ = stellar mass, $P$ = period, $e$ = orbital eccentricity.

### What can be measured from RV

* Planet mass (if transit is observed → true mass, as $i \approx 90^\circ$).  
* Constraints on orbital eccentricity.

### Limitations of RV

* Requires high-precision measurements (m/s or better).  
* Stellar activity noise (spots, granulation, oscillations).  
* Instrumental stability of spectrograph is critical.

---

## 3. Combined Methods

By combining both methods:

* Transit → planet radius.  
* RV → planet mass.  
* Together → density:

$$
\rho_p = \frac{M_p}{\tfrac{4}{3}\pi R_p^3}
$$

Density and temperature allow classification of the planet: gas giant, super-Earth, mini-Neptune, or rocky planet.


---
##  Resources  

### Scientific and Data Sources  
- **NASA Exoplanet Archive** – [https://exoplanetarchive.ipac.caltech.edu/](https://exoplanetarchive.ipac.caltech.edu/)  
  Primary database used for retrieving light curves, stellar parameters, and confirmed exoplanet data.  

- **TESS (Transiting Exoplanet Survey Satellite)** – [https://tess.mit.edu/](https://tess.mit.edu/)  
  Provides photometric observations used to detect planetary transits.  

- **ESO HARPS Archive** – [https://www.eso.org/sci/facilities/lasilla/instruments/harps.html](https://www.eso.org/sci/facilities/lasilla/instruments/harps.html)  
  Used for obtaining radial velocity measurements to confirm planetary candidates.  

- **ExoFOP-TESS** – [https://exofop.ipac.caltech.edu/tess/](https://exofop.ipac.caltech.edu/tess/)  
  Source of stellar and planetary parameters used for validation and cross-matching.  

- **MAST (Mikulski Archive for Space Telescopes)** – [https://mast.stsci.edu/](https://mast.stsci.edu/)  
  Provides access to TESS, Kepler, and Hubble mission data for astrophysical research.  

---

###  Machine Learning and Data Processing  
- **TensorFlow / PyTorch** – used for CNN model training and classification of light curves.  
- **Scikit-learn** – used for feature-based classifiers such as Random Forest.  
- **AstroPy** – for astronomical data analysis and manipulation.  
- **Lightkurve** – for downloading, cleaning, and visualizing light curves from TESS data.  
