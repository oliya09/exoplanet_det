# Exoplanet Detection via Transit and Radial Velocity Methods
## 1. Abstract

This project presents a novel, science-based platform for the automated detection and exploration of exoplanets using data from NASA missions such as **Kepler**, **TESS**, and **K2**. We developed a convolutional neural network (CNN) model trained on light curve data to identify planetary transits with high precision. The model processes data from the NASA Exoplanet Archive, applying preprocessing techniques such as NaN removal, normalization, and detrending to ensure robustness.

The system combines machine learning, astrophysics, and human-centered design to promote space science literacy, data accessibility, and public engagement. The project is fully documented, includes a development roadmap and business plan, and serves as a scalable model for scientific outreach powered by open data and AI.

## 2. The Transit Method

### 2.1 Overview

The **transit method** is based on observing periodic decreases in a star’s brightness caused by an orbiting planet passing in front of the stellar disk, as viewed from the observer’s line of sight.  
This temporary dimming, known as a **transit event**, occurs when the planet blocks part of the star’s emitted light, producing a small but measurable reduction in the observed flux.  
By analyzing these periodic brightness variations — known as **light curves** — key physical parameters of the planetary system can be determined.


<video width="640" height="360" controls>
  <source src="presentations/transit_method_multiple_planets_4K.mp4" type="video/mp4">
</video>


### 2.2 Transit Depth

When a planet transits its host star, it obscures a fraction of the stellar surface.  
If $F_0$ is the flux of the unobscured star and $F$ is the observed flux during the transit, the fractional decrease in brightness (the **transit depth**) is:

$$
\delta = \frac{F_0 - F}{F_0} \approx \left( \frac{R_p}{R_*} \right)^2
$$

where:  
- $R_p$ — planetary radius  
- $R_*$ — stellar radius  

This approximation assumes the planet is fully opaque and that **limb darkening** (gradual dimming of the stellar edge) is negligible.  
The measured transit depth provides a direct estimate of the planet-to-star radius ratio.  

For example:  
- A Jupiter-sized planet transiting a Sun-like star → $\delta \approx 1\%$ 
- An Earth-sized planet → $\delta \approx 0.01\%$



### 2.3 Orbital Period and Geometry

Each transit corresponds to one complete orbit of the planet.  
The time between consecutive transits defines the **orbital period** $P$:

$$
P = t_{n+1} - t_n
$$

where $t_n$ and $t_{n+1}$ are the times of successive transits.

Using **Kepler’s Third Law**, the orbital period relates to the semi-major axis $a$ as:

$$
a^3 = \frac{G M_* P^2}{4 \pi^2}
$$

where:  
- $G$ — gravitational constant  
- $M_*$ — stellar mass  

With $P$ and $M_*$ known, the orbital distance $a$ can be estimated, helping determine whether the planet lies within the **habitable zone**.



### 2.4 Transit Duration and Inclination

The **transit duration** $T_d$ — the time between the beginning and end of the flux dip — depends on the stellar radius, orbital distance, and orbital inclination $i$.  
For circular orbits:

$$
T_d \approx \frac{P R_*}{\pi a} \sqrt{1 - b^2}
$$

where:

$$
b = \frac{a \cos i}{R_*}
$$

is the **impact parameter**, representing the projected distance between the planet’s trajectory and the star’s center.  

- An **edge-on orbit** $( i \approx 90^\circ \ )$ → longer and deeper transit.  
- A **smaller inclination** may result in a partial or even undetectable transit.



### 2.5 Derived Planetary Parameters

From purely photometric data, several quantities can be derived:

| Parameter | Symbol | Derived From | Physical Meaning |
|------------|---------|--------------|------------------|
| Planetary radius | $R_p$ | $\delta = (R_p / R_*)^2$ | Planet size relative to the star |
| Orbital period | $P$ | Repetition of transits | Time for one complete orbit |
| Orbital distance | $a$ | Kepler’s 3rd law | Mean separation between star and planet |
| Orbital inclination |$i$ | Transit shape | Tilt of orbital plane |
| Transit duration | $T_d$ | Light curve width | Duration of the transit event |



### 2.6 Planetary Equilibrium Temperature (Optional)

Assuming **radiative equilibrium** between absorbed and emitted energy, the planet’s **equilibrium temperature** can be approximated as:

$$
T_{eq} = T_* \sqrt{\frac{R_*}{2a}} (1 - A)^{1/4}
$$

where:  
- $T_*$ — stellar effective temperature  
- $A$ — Bond albedo (fraction of reflected stellar light)

This provides an estimate of the planet’s **average thermal environment**, relevant for assessing potential habitability.



## 3. The Radial Velocity Method

### 3.1 Overview

The **radial velocity method** detects planets through the gravitational influence they exert on their host stars.  
A planet causes its star to move in a small orbit around their common **center of mass (barycenter)**.  
As the star moves toward and away from the observer, its light experiences a **Doppler shift** — blue-shifted when approaching and red-shifted when receding.  
These periodic shifts in the star’s spectral lines indicate the presence of an orbiting planet.

<video width="640" height="360" controls>
  <source src="presentations/radial_velocity-1.mp4" type="video/mp4">
</video>



### 3.2 Doppler Relation

The observed **radial velocity** $v_r$ of the star is related to the fractional wavelength change:

$$
\frac{\Delta \lambda}{\lambda_0} = \frac{v_r}{c}
$$

where:  
- $\Delta \lambda$ — wavelength change  
- $\lambda_0$ — rest wavelength  
- $v_r$ — radial velocity  
- $c$ — speed of light  

By measuring these shifts over time, astronomers construct a **radial velocity curve**, which encodes information about the planet’s orbit.


### 3.3 Stellar Motion and Momentum Conservation

Both the star and planet orbit their barycenter according to the **law of conservation of momentum**:

$$
M_* v_* = M_p v_p
$$

where:  
- $M_*$ — stellar mass  
- $M_p$ — planetary mass  
- $v_*$, $v_p$ — orbital velocities of the star and planet  

Because $M_* \gg M_p$, the star’s motion is small but **detectable** as a periodic velocity oscillation.



### 3.4 Radial Velocity Curve

The variation of the star’s radial velocity with time can be modeled as:

$$
v_r(t) = K [ \cos(\omega + \nu(t)) + e \cos(\omega) ]
$$

where:  
- $K$ — velocity semi-amplitude  
- $\omega$ — argument of periapsis  
- $\nu(t)$ — true anomaly (orbital phase angle)  
- $e$ — orbital eccentricity  

For circular orbits ( $e = 0$ ), this simplifies to a sinusoidal variation.



### 3.5 Planetary Mass Determination

The semi-amplitude $K$ is given by:

$$
K = \left( \frac{2 \pi G}{P} \right)^{1/3} 
\frac{M_p \sin i}{(M_* + M_p)^{2/3}} 
\frac{1}{\sqrt{1 - e^2}}
$$

where:  
- $P$ — orbital period  
- $i$ — orbital inclination  
- $G$ — gravitational constant  

Since $i$ is often unknown, the measurable quantity is $M_p \sin i$ — the **minimum mass**.  
When combined with the **transit method** (which provides $i$), the true planetary mass $M_p$ can be determined.


### 3.6 Extracted Information

From radial velocity data, one can derive:

| Parameter | Symbol | Derived From | Physical Meaning |
|------------|---------|--------------|------------------|
| Orbital period | $P$ | Repetition of velocity oscillations | Time for one orbit |
| Velocity amplitude | $K$ | Maximum Doppler shift | Strength of stellar motion |
| Eccentricity | $e$ | Shape of velocity curve | Orbit elongation |
| Orientation | $\omega$ | Curve phase | Orientation of orbit in space |
| Minimum mass | $M_p \sin i$ | Derived from $K, P, M_*$ | Lower limit of planet’s mass |

If combined with the transit method, the **true mass** and **mean density** of the planet can be calculated:

$$
\rho_p = \frac{3 M_p}{4 \pi R_p^3}
$$

This value helps distinguish **gas giants** from **rocky planets**.


## 4. Links to Resources

This project includes supporting materials that complement both the scientific and technical aspects of our work. The resources provide additional context, strategy, and presentation of our findings.  

You can view the resources at the following links:

- 🔗 [Business Plan](https://docs.google.com/document/d/1gx3jaj3RCqlsMDE43eFhbXgLpzUTG5RCYFpLTt2oMXA/edit?usp=sharing)

- 🔗 [Presentation](https://www.canva.com/design/DAG05C_frmI/G0zzf-tF-aK1GzcH6F_Jeg/edit?utm_content=DAG05C_frmI&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## 5. Referances

- NASA Exoplanet Archive – https://exoplanetarchive.ipac.caltech.edu  

- TESS Mission – https://tess.mit.edu  
- Kepler Mission – https://keplerscience.arc.nasa.gov  
- MAST Archive (Mikulski Archive for Space Telescopes) – https://mast.stsci.edu  
- ESO HARPS Radial Velocity Archive – https://www.eso.org/sci/facilities/lasilla/instruments/harps  
- ExoFOP-TESS (Exoplanet Follow-up Observing Program) – https://exofop.ipac.caltech.edu/tess  
- Lightkurve Python Package – https://docs.lightkurve.org  
- Astropy: The Astropy Project – https://www.astropy.org  
- TensorFlow – https://www.tensorflow.org  
- SIMBAD Astronomical Database – http://simbad.u-strasbg.fr/simbad/  
- Gaia Mission – https://gea.esac.esa.int/archive/ 
- Seager, S. & Mallén-Ornelas, G. (2003). “A Unique Solution of Planet and Star Parameters from an Extrasolar Planet Transit Light Curve.”  
     The Astrophysical Journal, 585(2), 1038–1055.  
- Winn, J. N. (2010). “Exoplanet Transits and Occultations.” In: Seager, S. (Ed.), Exoplanets. University of Arizona Press.



## 6. Scientific Impact

The use of transit and radial velocity data in conjunction provides the most complete picture of exoplanets.  
Our pipeline uses these NASA datasets to detect exoplanets automatically, contributing to astronomical research and democratizing access to space science through data-driven platforms.