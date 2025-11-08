from fastapi import FastAPI, APIRouter, HTTPException, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import httpx
import asyncio
import json
import math
from io import BytesIO
import base64
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import colorsys

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== MODELS ==========

class ExoplanetSearchRequest(BaseModel):
    query: str
    search_type: str = "name"  # "name" or "tic"

class HabitabilityScore(BaseModel):
    total_score: float  # 0-100
    survival_chance: float  # 0-100%
    factors: Dict[str, Any]
    category: str  # "Hostile", "Challenging", "Moderate", "Promising", "Earth-like"
    recommendations: List[str]
    risks: List[str]

class ExoplanetData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    tic_id: Optional[str] = None
    nasa_data: Optional[Dict[str, Any]] = None
    simbad_data: Optional[Dict[str, Any]] = None
    physical_params: Optional[Dict[str, Any]] = None
    orbital_params: Optional[Dict[str, Any]] = None
    host_star: Optional[Dict[str, Any]] = None
    discovery_info: Optional[Dict[str, Any]] = None
    habitability_score: Optional[HabitabilityScore] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AIAnalysisRequest(BaseModel):
    exoplanet_data: Dict[str, Any]
    ollama_url: str = "http://localhost:11434"

class AIAnalysisResponse(BaseModel):
    analysis: str
    ollama_available: bool
    error: Optional[str] = None

# CORS configuration
origins = [
    "http://localhost:3000",
    "https://penalties-sometimes-ago-understand.trycloudflare.com",
    "https://claims-keeping-mid-friend.trycloudflare.com",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== HABITABILITY SCORING SYSTEM ==========

def calculate_habitability_score(physical_params: dict, orbital_params: dict, host_star: dict) -> HabitabilityScore:
    """
    Advanced habitability scoring system based on multiple factors
    Hybrid approach combining ESI (Earth Similarity Index) and custom survival metrics
    """
    scores = {}
    factors = {}
    risks = []
    recommendations = []
    
    # 1. Temperature Score (0-25 points)
    temp = physical_params.get('equilibrium_temp')
    if temp:
        if 273 <= temp <= 373:  # Water liquid range
            temp_score = 25
            factors['temperature'] = {'score': 25, 'value': temp, 'status': 'Optimal'}
        elif 200 <= temp <= 400:
            temp_score = 15
            factors['temperature'] = {'score': 15, 'value': temp, 'status': 'Survivable with protection'}
            risks.append(f"Temperature {temp}K requires environmental suits")
        elif 150 <= temp <= 500:
            temp_score = 5
            factors['temperature'] = {'score': 5, 'value': temp, 'status': 'Extreme'}
            risks.append(f"Extreme temperature {temp}K - high risk")
        else:
            temp_score = 0
            factors['temperature'] = {'score': 0, 'value': temp, 'status': 'Lethal'}
            risks.append(f"Lethal temperature {temp}K")
    else:
        temp_score = 10
        factors['temperature'] = {'score': 10, 'value': None, 'status': 'Unknown'}
    
    scores['temperature'] = temp_score
    



    # 2. Mass/Gravity Score (0-20 points)
    mass_earth = physical_params.get('pl_bmasse')
    mass_jupiter = physical_params.get('pl_bmassj')

    if mass_earth is not None:
        mass = float(mass_earth)
    elif mass_jupiter is not None:
        mass = float(mass_jupiter) * 317.8
    else:
        mass = None

    radius_earth = physical_params.get('pl_rade')
    radius_jupiter = physical_params.get('pl_radj')

    if radius_earth is not None:
        radius = float(radius_earth)
    elif radius_jupiter is not None:
        radius = float(radius_jupiter) * 11.2  # 1 R_Jup = 11.2 R_Earth
    else:
        radius = None

    if mass is not None and mass > 0:
        if radius is not None and radius > 0:
            gravity_ratio = mass / (radius ** 2)  # Proper surface gravity (Earth=1)
        else:
            gravity_ratio = mass  # Fallback if radius unknown

        if 0.5 <= gravity_ratio <= 2.0:
            mass_score = 20
            factors['gravity'] = {'score': 20, 'gravity_g': round(gravity_ratio, 2), 'status': 'Comfortable'}
        elif 0.3 <= gravity_ratio <= 3.0:
            mass_score = 12
            factors['gravity'] = {'score': 12, 'gravity_g': round(gravity_ratio, 2), 'status': 'Adaptable'}
            recommendations.append("Gravity adaptation training required")
        elif gravity_ratio > 3.0:
            mass_score = 3
            factors['gravity'] = {'score': 3, 'gravity_g': round(gravity_ratio, 2), 'status': 'Crushing'}
            risks.append(f"High gravity ({gravity_ratio:.1f}x Earth) - extreme physical stress")
        else:
            mass_score = 5
            factors['gravity'] = {'score': 5, 'gravity_g': round(gravity_ratio, 2), 'status': 'Low'}
            risks.append("Low gravity - long-term health risks")
    else:
        mass_score = 10
        factors['gravity'] = {'score': 10, 'gravity_g': None, 'status': 'Unknown'}

    scores['gravity'] = mass_score

    
    
    # 3. Stellar Radiation (0-20 points)
    star_temp = host_star.get('temperature')
    star_radius = host_star.get('radius')
    distance = orbital_params.get('semi_major_axis')

    if star_temp and star_radius and distance and distance > 0:
        try:
            luminosity = (star_temp / 5778) ** 4 * (star_radius ** 2)
            received_radiation = luminosity / (distance ** 2)

            if 0.8 <= received_radiation <= 1.2:
                radiation_score = 20
                factors['radiation'] = {'score': 20, 'relative_to_earth': round(received_radiation, 3), 'status': 'Safe'}
            elif 0.5 <= received_radiation <= 1.8:
                radiation_score = 12
                factors['radiation'] = {'score': 12, 'relative_to_earth': round(received_radiation, 3), 'status': 'Moderate'}
                recommendations.append("Enhanced UV protection needed")
            elif received_radiation > 2.0:
                radiation_score = 3
                factors['radiation'] = {'score': 3, 'relative_to_earth': round(received_radiation, 3), 'status': 'Dangerous'}
                risks.append("High radiation levels - shielding critical")
            else:
                radiation_score = 8
                factors['radiation'] = {'score': 8, 'relative_to_earth': round(received_radiation, 3), 'status': 'Low'}
                risks.append("Insufficientstellar energy - heating required")
        except:
            radiation_score = 10
            factors['radiation'] = {'score': 10, 'relative_to_earth': None, 'status': 'Calculation error'}
    else:
        radiation_score = 10
        factors['radiation'] = {'score': 10, 'relative_to_earth': None, 'status': 'Unknown'}

    scores['radiation'] = radiation_score
    # 4. Orbital Stability (0-15 points)
    eccentricity = orbital_params.get('eccentricity')
    if eccentricity is not None:
        if eccentricity < 0.1:
            orbital_score = 15
            factors['orbit'] = {'score': 15, 'eccentricity': eccentricity, 'status': 'Stable'}
        elif eccentricity < 0.3:
            orbital_score = 10
            factors['orbit'] = {'score': 10, 'eccentricity': eccentricity, 'status': 'Moderate variation'}
            recommendations.append("Seasonal temperature variations expected")
        else:
            orbital_score = 3
            factors['orbit'] = {'score': 3, 'eccentricity': eccentricity, 'status': 'Highly elliptical'}
            risks.append(f"High orbital eccentricity ({eccentricity}) - extreme seasons")
    else:
        orbital_score = 8
        factors['orbit'] = {'score': 8, 'eccentricity': None, 'status': 'Unknown'}
    
    scores['orbital_stability'] = orbital_score
    
    # 5. Planet Type/Composition (0-20 points)
    if radius is not None and mass is not None and mass > 0:
        density = mass / (radius ** 3) if radius > 0 else 0
        
        if 0.8 <= radius <= 1.5 and 0.5 <= mass <= 2.0:
            type_score = 20
            factors['type'] = {'score': 20, 'classification': 'Rocky/Terrestrial', 'status': 'Earth-like'}
        elif radius < 2.0 and mass < 10:
            type_score = 12
            factors['type'] = {'score': 12, 'classification': 'Super-Earth', 'status': 'Potentially habitable'}
            recommendations.append("Super-Earth environment - enhanced structural support needed")
        elif radius > 3.0 or mass > 50:
            type_score = 1
            factors['type'] = {'score': 1, 'classification': 'Gas Giant', 'status': 'No solid surface'}
            risks.append("Gas giant - no solid surface for habitation")
        else:
            type_score = 8
            factors['type'] = {'score': 8, 'classification': 'Uncertain', 'status': 'Unknown composition'}
    else:
        type_score = 10
        factors['type'] = {'score': 10, 'classification': 'Unknown', 'status': 'Insufficient data'}
    
    scores['planet_type'] = type_score
    
    # Calculate total score
    total_score = sum(scores.values())
    
    # Calculate survival chance (non-linear scaling)
    if total_score >= 85:
        survival_chance = 95 + (total_score - 85) / 15 * 5
        category = "Earth-like"
        recommendations.append("Excellent colonization candidate!")
    elif total_score >= 70:
        survival_chance = 75 + (total_score - 70) / 15 * 20
        category = "Promising"
        recommendations.append("Viable with standard equipment")
    elif total_score >= 50:
        survival_chance = 45 + (total_score - 50) / 20 * 30
        category = "Moderate"
        recommendations.append("Advanced life support required")
    elif total_score >= 30:
        survival_chance = 15 + (total_score - 30) / 20 * 30
        category = "Challenging"
        recommendations.append("Extreme survival conditions - expert team only")
    else:
        survival_chance = max(1, total_score / 30 * 15)
        category = "Hostile"
        risks.append("Extremely hostile environment - not recommended for colonization")
    
    # Add general risks
    if not physical_params.get('density'):
        risks.append("Unknown atmospheric composition")
    
    return HabitabilityScore(
        total_score=round(total_score, 2),
        survival_chance=round(survival_chance, 2),
        factors=factors,
        category=category,
        recommendations=recommendations,
        risks=risks
    )

# ========== REAL IMAGE FETCHING ==========

async def fetch_real_exoplanet_image(planet_name: str, host_star: dict) -> Dict[str, Any]:
    """
    Fetch real images from multiple sources:
    1. NASA's Eyes on Exoplanets
    2. ESA Hubble
    3. JWST (if available)
    4. Ground-based observatories
    """
    images = {
        'real_images': [],
        'synthetic_image': None,
        'hubble_data': None,
        'jwst_data': None
    }
    
    try:
        # 1. Try NASA Eyes on Exoplanets API
        eyes_data = await fetch_nasa_eyes_data(planet_name)
        if eyes_data:
            images['eyes_on_exoplanets'] = eyes_data
        
        # 2. Try ESA/Hubble
        hubble_image = await fetch_hubble_image(host_star.get('name', planet_name))
        if hubble_image:
            images['hubble_data'] = hubble_image
            images['real_images'].append({
                'source': 'Hubble Space Telescope',
                'url': hubble_image.get('url'),
                'description': 'Host star observation'
            })
        
        # 3. Try JWST Archive
        jwst_image = await fetch_jwst_image(planet_name)
        if jwst_image:
            images['jwst_data'] = jwst_image
            images['real_images'].append({
                'source': 'James Webb Space Telescope',
                'url': jwst_image.get('url'),
                'description': 'Recent observation'
            })
        
        # 4. Sloan Digital Sky Survey
        sdss_image = await fetch_sdss_image(host_star)
        if sdss_image:
            images['real_images'].append({
                'source': 'SDSS',
                'url': sdss_image,
                'description': 'Wide-field observation'
            })
        
        # 5. PanSTARRS
        panstarrs_image = await fetch_panstarrs_image(host_star)
        if panstarrs_image:
            images['real_images'].append({
                'source': 'PanSTARRS',
                'url': panstarrs_image,
                'description': 'Deep sky survey'
            })

        # 6. NASA SkyView (2MASS)
        skyview_img = await fetch_skyview_image(host_star)
        if skyview_img:
            images['real_images'].append({
                'source': 'NASA SkyView (2MASS)',
                'url': skyview_img,  # Уже base64!
                'description': 'Real infrared image of host star field'
            })
        
    except Exception as e:
        logger.error(f"Error fetching real images: {str(e)}")
    
    return images

async def fetch_nasa_eyes_data(planet_name: str) -> Optional[Dict[str, Any]]:
    """Fetch data from NASA's Eyes on Exoplanets"""
    try:
        # NASA Eyes on Exoplanets uses data from Exoplanet Archive
        # We can construct a deep link to their web app
        base_url = "https://eyes.nasa.gov/apps/exo/"
        
        # Create a reference link
        eyes_link = f"{base_url}#/planet/{planet_name.replace(' ', '_')}"
        
        return {
            'available': True,
            'link': eyes_link,
            'description': 'Interactive 3D visualization available at NASA Eyes on Exoplanets',
            'features': [
                'Real-time 3D model',
                'Orbital animation',
                'Comparative views',
                'System exploration'
            ]
        }
    except Exception as e:
        logger.error(f"NASA Eyes error: {str(e)}")
        return None

async def fetch_hubble_image(star_name: str):
    try:
        url = "https://hla.stsci.edu/cgi-bin/hlaSIAP.cgi"
        params = {
            'target': star_name,
            'format': 'json',
            'imagetype': 'best'
        }
       
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url, params=params)
            
            if not response.text.strip().startswith("["):
                logger.info("Hubble returned non-JSON, skipping.")
                return None
                
            data = response.json()
            if data and len(data) > 0:
                return {'url': data[0].get('AccessReference'), 'available': True}
            return None
    except:
        return None

async def fetch_jwst_image(planet_name: str):
    try:
        search_url = "https://mast.stsci.edu/api/v0.1/search"
        params = {
            'q': planet_name,
            't': 'observations',
            'mission': 'JWST'
        }
        async with httpx.AsyncClient() as client:
            r = await client.get(search_url, params=params)
            if r.status_code == 200 and r.json().get('results'):
                return {'available': True, 'note': 'JWST observations exist'}
        return None
    except:
        return None

async def fetch_sdss_image(host_star: dict) -> Optional[str]:
    """Fetch image from Sloan Digital Sky Survey"""
    try:
        ra = host_star.get('coordinates', {}).get('ra')
        dec = host_star.get('coordinates', {}).get('dec')
        
        if not ra or not dec:
            return None
        
        # SDSS Image Cutout Service
        base_url = "https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg"
        params = {
            'ra': ra,
            'dec': dec,
            'scale': 0.4,
            'width': 512,
            'height': 512
        }
        
        url = f"{base_url}?ra={ra}&dec={dec}&scale=0.4&width=512&height=512"
        
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.head(url)
            if response.status_code == 200:
                logger.info("SDSS: Image available")
                return url
        
        return None
    except Exception as e:
        logger.error(f"SDSS error: {str(e)}")
        return None

async def fetch_panstarrs_image(host_star: dict) -> Optional[str]:
    """Fetch image from PanSTARRS"""
    try:
        ra = host_star.get('coordinates', {}).get('ra')
        dec = host_star.get('coordinates', {}).get('dec')
        
        if not ra or not dec:
            return None
        
        # PanSTARRS Image Cutout Service
        url = f"https://ps1images.stsci.edu/cgi-bin/ps1cutouts?pos={ra}+{dec}&filter=color&size=240"
        
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.head(url)
            if response.status_code == 200:
                logger.info("PanSTARRS: Image available")
                return url
        
        return None
    except Exception as e:
        logger.error(f"PanSTARRS error: {str(e)}")
        return None




# ..........................................
async def fetch_skyview_image(host_star: dict) -> Optional[str]:
    """Fetch real sky image from NASA SkyView for host star"""
    try:
        ra = host_star.get('coordinates', {}).get('ra')  # From SIMBAD or NASA
        dec = host_star.get('coordinates', {}).get('dec')
        if not ra or not dec:
            # Fallback: search coords for star name
            star_name = host_star.get('name', '')
            if star_name:
                coord_url = f"https://simbad.cds.unistra.fr/simbad/sim-basic?Ident={star_name}&output.format=json"
                async with httpx.AsyncClient() as client:
                    resp = await client.get(coord_url)
                    if resp.status_code == 200:
                        data = resp.json()
                        ra = data.get('RA', [None])[0]
                        dec = data.get('DEC', [None])[0]
        
        if ra and dec:
            # SkyView API: survey=2MASS, size=0.5 deg
            skyview_url = f"https://skyview.gsfc.nasa.gov/current/cgi/query.pl?survey=2MASS-J&project=IPAC&coordinates={ra}+{dec}&pixels=512&size=0.5"
            
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(skyview_url)
                if response.status_code == 200:
                    # Download FITS and convert to PNG (use astropy if installed, or simple fetch)
                    # For simplicity: assume direct PNG link or base64
                    img_data = response.content
                    img = Image.open(BytesIO(img_data)).convert('RGB')
                    buffer = BytesIO()
                    img.save(buffer, format='PNG')
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    logger.info("SkyView: Image embedded")
                    return f"data:image/png;base64,{img_str}"
        return None
    except Exception as e:
        logger.error(f"SkyView error: {str(e)}")
        return None
    


# ========== NASA EXOPLANET ARCHIVE API ==========

async def fetch_nasa_exoplanet_data(query: str, search_type: str = "name"):
    """Fetch data from NASA Exoplanet Archive"""
    try:
        base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        
        if search_type == "tic":
            adql_query = f"""
            SELECT * FROM ps 
            WHERE tic_id = '{query}'
            """
        else:
            adql_query = f"""
            SELECT * FROM ps 
            WHERE LOWER(pl_name) LIKE LOWER('%{query}%') 
            OR LOWER(hostname) LIKE LOWER('%{query}%')
            """
        
        params = {
            "query": adql_query,
            "format": "json"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data:
                logger.info(f"NASA: Found {len(data)} results")
                return data[0] if isinstance(data, list) and len(data) > 0 else data
            return None
            
    except Exception as e:
        logger.error(f"NASA API error: {str(e)}")
        return None

# ========== SIMBAD API ==========

async def fetch_simbad_data(object_name: str):
    try:
        base_url = "https://simbad.cds.unistra.fr/simbad/sim-id"
        params = {
            "Ident": object_name.replace(" ", "+"),
            "output.format": "json"
        }
       
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(base_url, params=params)
            
            # Проверка на HTML
            if response.text.strip().startswith("<"):
                logger.info("SIMBAD returned HTML (likely login page), skipping.")
                return None
                
            response.raise_for_status()
            return response.json()
            
    except Exception as e:
        logger.warning(f"SIMBAD failed: {str(e)}")
        return None

# ========== DATA AGGREGATION ==========

async def aggregate_exoplanet_data(query: str, search_type: str = "name"):
    """Aggregate data from all sources"""
    logger.info(f"Searching for: {query} (type: {search_type})")
    
    tic_id = None
    if search_type == "tic":
        tic_id = query.replace('TIC', '').replace('TIC ', '').strip()
        query = f"TIC {tic_id}"
    
    nasa_task = fetch_nasa_exoplanet_data(query, search_type)
    simbad_task = fetch_simbad_data(query)
    
    results = await asyncio.gather(nasa_task, simbad_task, return_exceptions=True)
    
    nasa_data = results[0] if not isinstance(results[0], Exception) else None
    simbad_data = results[1] if not isinstance(results[1], Exception) else None
    
    if not any([nasa_data, simbad_data]):
        return None
    
    physical_params = extract_physical_params(nasa_data)
    orbital_params = extract_orbital_params(nasa_data)
    host_star = extract_host_star_data(nasa_data, simbad_data)
    
    result = {
        "name": query,
        "nasa_data": nasa_data,
        "simbad_data": simbad_data,
        "physical_params": physical_params,
        "orbital_params": orbital_params,
        "host_star": host_star,
        "discovery_info": extract_discovery_info(nasa_data),
        "tic_id": nasa_data.get('tic_id') if nasa_data else tic_id,
        "sources_used": ["NASA Exoplanet Archive"] if nasa_data else []
    }
    
    if simbad_data:
        result["sources_used"].append("SIMBAD")
    
    return result

def extract_physical_params(nasa_data):
    """Extract physical parameters"""
    params = {}
    
    if nasa_data:
        params.update({
            "mass": nasa_data.get('pl_bmasse'),
            "mass_jupiter": nasa_data.get('pl_bmassj'),
            "radius": nasa_data.get('pl_rade'),
            "radius_jupiter": nasa_data.get('pl_radj'),
            "density": nasa_data.get('pl_dens'),
            "gravity": nasa_data.get('pl_g'),
            "equilibrium_temp": nasa_data.get('pl_eqt'),
        })
    
    return params

def extract_orbital_params(nasa_data):
    """Extract orbital parameters"""
    params = {}
    
    if nasa_data:
        params.update({
            "period": nasa_data.get('pl_orbper'),
            "semi_major_axis": nasa_data.get('pl_orbsmax'),
            "eccentricity": nasa_data.get('pl_orbeccen'),
            "inclination": nasa_data.get('pl_orbincl'),
            "periastron": nasa_data.get('pl_orblper'),
        })
    
    return params



def extract_host_star_data(nasa_data, simbad_data):
    """Extract host star data"""
    star_data = {}
    
    if nasa_data:
        star_data.update({
            "name": nasa_data.get('hostname'),
            "mass": nasa_data.get('st_mass'),
            "radius": nasa_data.get('st_rad'),
            "temperature": nasa_data.get('st_teff'),
            "luminosity": nasa_data.get('st_lum'),
            "metallicity": nasa_data.get('st_met'),
            "age": nasa_data.get('st_age'),
            "distance": nasa_data.get('sy_dist'),
            "spectral_type": nasa_data.get('st_spectype'),
        })
    
    if simbad_data:
        star_data.update({
          "simbad_type": simbad_data.get('otype'),
            "coordinates": simbad_data.get('coord'),
        })
    
    return star_data

def extract_discovery_info(nasa_data):
    """Extract discovery information"""
    info = {}
    
    if nasa_data:
        info.update({
            "discovery_method": nasa_data.get('discoverymethod'),
            "discovery_year": nasa_data.get('disc_year'),
            "discovery_facility": nasa_data.get('disc_facility'),
            "discovery_telescope": nasa_data.get('disc_telescope'),
        })
    
    return info

# ========== ENHANCED IMAGE GENERATION ==========

def generate_enhanced_star_image(star_data: dict, planet_name: str = "") -> str:
    """Generate enhanced synthetic star image with better effects"""
    try:
        size = 600
        img = Image.new('RGB', (size, size), color='#000008')
        draw = ImageDraw.Draw(img)
        
        temp = star_data.get('temperature', 5778)
        
        # More accurate color based on blackbody radiation
        if temp > 30000:
            color = (155, 176, 255)
        elif temp > 10000:
            color = (202, 215, 255)
        elif temp > 7500:
            color = (248, 247, 255)
        elif temp > 6000:
            color = (255, 244, 234)
        elif temp > 5200:
            color = (255, 237, 210)
        elif temp > 3700:
            color = (255, 204, 111)
        else:
            color = (255, 121, 63)
        
        center_x, center_y = size // 2, size // 2
        star_radius = int((star_data.get('radius', 1) * 100) if star_data.get('radius', 1) < 3 else 180)
        
        # Draw starfield background
        np.random.seed(42)
        for _ in range(200):
            x = np.random.randint(0, size)
            y = np.random.randint(0, size)
            brightness = np.random.randint(50, 200)
            draw.ellipse([x-1, y-1, x+1, y+1], fill=(brightness, brightness, brightness))
        
        # Enhanced glow with multiple layers
        for i in range(8, 0, -1):
            glow_radius = star_radius + i * 25
            alpha = int(80 / (i * 0.8))
            glow_color = tuple(int(c * (1 - i * 0.08)) for c in color)
            draw.ellipse(
                [center_x - glow_radius, center_y - glow_radius,
                 center_x + glow_radius, center_y + glow_radius],
                fill=glow_color
            )
        
        # Main star with gradient effect
        draw.ellipse(
            [center_x - star_radius, center_y - star_radius,
             center_x + star_radius, center_y + star_radius],
            fill=color
        )
        
        # Core highlight
        core_radius = star_radius // 3
        core_color = tuple(min(255, int(c * 1.3)) for c in color)
        draw.ellipse(
            [center_x - core_radius, center_y - core_radius,
             center_x + core_radius, center_y + core_radius],
            fill=core_color
        )
        
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        
        # Add info overlay
        draw = ImageDraw.Draw(img)
        info_text = f"{star_data.get('name', 'Unknown Star')}\n{temp}K"
        draw.text((10, 10), info_text, fill='white')
        
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        logger.error(f"Error generating enhanced star image: {str(e)}")
        return None

def generate_habitability_visualization(habitability: HabitabilityScore, planet_name: str) -> str:
    """Generate visual habitability score dashboard"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10), facecolor='#0a0e27')
        fig.suptitle(f'Habitability Analysis: {planet_name}', 
                     color='white', fontsize=16, fontweight='bold')
        
        # 1. Overall Score Gauge
        ax1.set_facecolor('#0a0e27')
        score = habitability.total_score
        
        # Draw gauge
        theta = np.linspace(np.pi, 0, 100)
        r = 1
        
        # Background arc
        ax1.plot(r * np.cos(theta), r * np.sin(theta), 'gray', linewidth=20, alpha=0.3)
        
        # Score arc with color gradient
        score_theta = np.linspace(np.pi, np.pi * (1 - score/100), int(score))
        if score >= 70:
            color = '#00ff00'
        elif score >= 50:
            color = '#ffff00'
        elif score >= 30:
            color = '#ff8800'
        else:
            color = '#ff0000'
        
        ax1.plot(r * np.cos(score_theta), r * np.sin(score_theta), color, linewidth=20)
        
        # Add score text
        ax1.text(0, -0.3, f'{score:.1f}/100', ha='center', va='center', 
                fontsize=24, color='white', fontweight='bold')
        ax1.text(0, -0.5, habitability.category, ha='center', va='center', 
                fontsize=14, color=color)
        
        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1, 1)
        ax1.axis('off')
        ax1.set_title('Overall Habitability Score', color='white', fontsize=12, pad=10)
        
        # 2. Survival Chance
        ax2.set_facecolor('#0a0e27')
        survival = habitability.survival_chance
        
        wedge = Wedge((0, 0), 1, 0, survival * 3.6, facecolor=color, edgecolor='white', linewidth=2)
        ax2.add_patch(wedge)
        
        circle = Circle((0, 0), 0.7, facecolor='#0a0e27', edgecolor='none')
        ax2.add_patch(circle)
        
        ax2.text(0, 0, f'{survival:.1f}%', ha='center', va='center', 
                fontsize=24, color='white', fontweight='bold')
        ax2.text(0, -0.3, 'Survival Rate', ha='center', va='center', 
                fontsize=10, color='white')
        
        ax2.set_xlim(-1.2, 1.2)
        ax2.set_ylim(-1.2, 1.2)
        ax2.axis('off')
        ax2.set_title('Estimated Survival Chance', color='white', fontsize=12, pad=10)
        
        # 3. Factor Breakdown
        ax3.set_facecolor('#0a0e27')
        factors = habitability.factors
        
        categories = []
        scores = []
        max_scores = {'temperature': 25, 'gravity': 20, 'radiation': 20, 
                     'orbit': 15, 'type': 20}
        
        for key, data in factors.items():
            if isinstance(data, dict) and 'score' in data:
                categories.append(key.capitalize())
                scores.append(data['score'])
        
        y_pos = np.arange(len(categories))
        colors_bar = ['#00ff00' if s/max_scores.get(k.lower(), 20) > 0.7 
                     else '#ffff00' if s/max_scores.get(k.lower(), 20) > 0.5
                     else '#ff8800' if s/max_scores.get(k.lower(), 20) > 0.3
                     else '#ff0000' 
                     for s, k in zip(scores, categories)]
        
        ax3.barh(y_pos, scores, color=colors_bar, alpha=0.8, edgecolor='white')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(categories, color='white')
        ax3.set_xlabel('Score', color='white')
        ax3.set_title('Factor Breakdown', color='white', fontsize=12, pad=10)
        ax3.tick_params(colors='white')
        ax3.grid(axis='x', alpha=0.2, color='white')
        ax3.spines['bottom'].set_color('white')
        ax3.spines['left'].set_color('white')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # 4. Risks and Recommendations
        ax4.set_facecolor('#0a0e27')
        ax4.axis('off')
        
        text_content = "⚠️  KEY RISKS:\n"
        for i, risk in enumerate(habitability.risks[:3], 1):
            text_content += f"{i}. {risk}\n"
        
        text_content += "\n✓  RECOMMENDATIONS:\n"
        for i, rec in enumerate(habitability.recommendations[:3], 1):
            text_content += f"{i}. {rec}\n"
        
        ax4.text(0.05, 0.95, text_content, transform=ax4.transAxes,
                fontsize=9, color='white', verticalalignment='top',
                family='monospace', bbox=dict(boxstyle='round', 
                facecolor='#1a1e3a', alpha=0.8, edgecolor='#00ffff'))
        
        ax4.set_title('Mission Planning', color='white', fontsize=12, pad=10)
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor='#0a0e27', dpi=120, bbox_inches='tight')
        plt.close()
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        logger.error(f"Error generating habitability visualization: {str(e)}")
        return None

def generate_orbital_diagram(orbital_params: dict, planet_name: str) -> str:
    """Generate enhanced orbital diagram"""
    try:
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0a0e27')
        ax.set_facecolor('#0a0e27')
        
        semi_major = orbital_params.get('semi_major_axis', 1.0)
        eccentricity = orbital_params.get('eccentricity', 0.0)
        
        if semi_major is None:
            semi_major = 1.0
        if eccentricity is None:
            eccentricity = 0.0
        
        # Calculate orbital points
        theta = np.linspace(0, 2 * np.pi, 200)
        r = semi_major * (1 - eccentricity**2) / (1 + eccentricity * np.cos(theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Plot orbit with gradient
        for i in range(len(x)-1):
            alpha = 0.3 + 0.4 * (i / len(x))
            ax.plot(x[i:i+2], y[i:i+2], color='#00ffff', linewidth=2, alpha=alpha)
        
        # Plot star at focus
        ax.plot(0, 0, 'o', color='#ffff00', markersize=25, 
               markeredgecolor='white', markeredgewidth=2, label='Star', zorder=10)
        
        # Plot planet at current position
        planet_x = r[0] * np.cos(0)
        planet_y = r[0] * np.sin(0)
        ax.plot(planet_x, planet_y, 'o', color='#00aaff', markersize=15, 
               markeredgecolor='white', markeredgewidth=2, label='Planet', zorder=10)
        
        # Draw velocity vector
        ax.arrow(planet_x, planet_y, 0, 0.3, head_width=0.1, 
                head_length=0.1, fc='cyan', ec='cyan', alpha=0.6)
        
        # Add habitable zone
        habitable_inner = 0.95
        habitable_outer = 1.37
        
        theta_hab = np.linspace(0, 2*np.pi, 100)
        ax.fill_between(habitable_outer * np.cos(theta_hab), 
                        habitable_outer * np.sin(theta_hab),
                        habitable_inner * np.cos(theta_hab),
                        alpha=0.2, color='green', label='Habitable Zone')
        
        # Add distance markers
        for dist in [0.5, 1.0, 1.5, 2.0]:
            circle = Circle((0, 0), dist, fill=False, 
                          linestyle='--', alpha=0.2, color='white', linewidth=1)
            ax.add_patch(circle)
            ax.text(dist, 0, f'{dist} AU', color='white', 
                   fontsize=8, ha='left', va='bottom')
        
        # Add perihelion and aphelion
        perihelion = semi_major * (1 - eccentricity)
        aphelion = semi_major * (1 + eccentricity)
        ax.plot(perihelion, 0, 'x', color='red', markersize=10, markeredgewidth=2)
        ax.plot(-aphelion, 0, 'x', color='blue', markersize=10, markeredgewidth=2)
        ax.text(perihelion, -0.2, 'Perihelion', color='red', fontsize=8, ha='center')
        ax.text(-aphelion, -0.2, 'Aphelion', color='blue', fontsize=8, ha='center')
        
        ax.set_xlabel('Distance (AU)', color='white', fontsize=12)
        ax.set_ylabel('Distance (AU)', color='white', fontsize=12)
        ax.set_title(f'Orbital Diagram: {planet_name}\n'
                    f'Period: {orbital_params.get("period", "?")} days | '
                    f'Eccentricity: {eccentricity:.3f}', 
                    color='white', fontsize=14, pad=20)
        ax.legend(loc='upper right', facecolor='#1a1e3a', 
                 edgecolor='#00ffff', labelcolor='white')
        ax.grid(True, alpha=0.2, color='white', linestyle=':')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        
        max_dist = max(aphelion, habitable_outer) * 1.3
        ax.set_xlim(-max_dist, max_dist)
        ax.set_ylim(-max_dist, max_dist)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor='#0a0e27', dpi=120, bbox_inches='tight')
        plt.close()
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        logger.error(f"Error generating orbital diagram: {str(e)}")
        return None

def generate_mass_radius_chart(physical_params: dict, planet_name: str) -> str:
    """Generate enhanced mass vs radius comparison chart"""
    try:
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0a0e27')
        ax.set_facecolor('#0a0e27')
        
        # Reference planets with colors
        reference_planets = [
            {'name': 'Mercury', 'mass': 0.055, 'radius': 0.383, 'color': '#8C7853'},
            {'name': 'Venus', 'mass': 0.815, 'radius': 0.949, 'color': '#FFC649'},
            {'name': 'Earth', 'mass': 1.0, 'radius': 1.0, 'color': '#4169E1'},
            {'name': 'Mars', 'mass': 0.107, 'radius': 0.532, 'color': '#CD5C5C'},
            {'name': 'Jupiter', 'mass': 317.8, 'radius': 11.2, 'color': '#DAA520'},
            {'name': 'Saturn', 'mass': 95.2, 'radius': 9.4, 'color': '#F4A460'},
            {'name': 'Uranus', 'mass': 14.5, 'radius': 4.0, 'color': '#4FD0E0'},
            {'name': 'Neptune', 'mass': 17.1, 'radius': 3.9, 'color': '#4169E1'},
        ]
        
        # Plot reference planets
        for planet in reference_planets:
            ax.scatter(planet['mass'], planet['radius'], s=150, alpha=0.6, 
                      color=planet['color'], edgecolors='white', linewidth=2,
                      label=planet['name'])
            ax.text(planet['mass'], planet['radius'], planet['name'], 
                   fontsize=8, color='white', ha='right', va='bottom')
        
        # Plot target planet
        target_mass = physical_params.get('mass', 1.0)
        target_radius = physical_params.get('radius', 1.0)
        
        if target_mass and target_radius:
            ax.scatter(target_mass, target_radius, s=400, color='#00ffff', 
                      marker='*', edgecolors='white', linewidth=3, 
                      label=planet_name, zorder=100)
            
            # Add connecting line to Earth for comparison
            ax.plot([1.0, target_mass], [1.0, target_radius], 
                   'w--', alpha=0.5, linewidth=1)
            
            # Calculate density
            if target_mass > 0 and target_radius > 0:
                density = target_mass / (target_radius ** 3)
                earth_density = 1.0
                density_ratio = density / earth_density
                
                ax.text(target_mass, target_radius * 1.1, 
                       f'{planet_name}\n{density_ratio:.2f}x Earth density', 
                       fontsize=9, color='cyan', ha='center', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='#1a1e3a', alpha=0.8))
        
        # Add composition regions
        ax.axhspan(0.3, 2.0, alpha=0.1, color='brown', label='Rocky/Terrestrial')
        ax.axhspan(2.0, 6.0, alpha=0.1, color='blue', label='Ice Giant')
        ax.axhspan(6.0, 20.0, alpha=0.1, color='orange', label='Gas Giant')
        
        ax.set_xlabel('Mass (Earth Masses)', color='white', fontsize=12, fontweight='bold')
        ax.set_ylabel('Radius (Earth Radii)', color='white', fontsize=12, fontweight='bold')
        ax.set_title('Mass vs Radius: Planet Classification', 
                    color='white', fontsize=14, pad=20, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc='upper left', facecolor='#1a1e3a', edgecolor='#00ffff', 
                 labelcolor='white', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, color='white', which='both', linestyle=':')
        ax.tick_params(colors='white')
        
        for spine in ax.spines.values():
            spine.set_color('white')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor='#0a0e27', dpi=120, bbox_inches='tight')
        plt.close()
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        logger.error(f"Error generating mass-radius chart: {str(e)}")
        return None

# ========== OLLAMA AI ANALYSIS ==========

async def analyze_with_ollama(data: dict, ollama_url: str = "http://localhost:11434"):
    """Enhanced AI analysis with habitability context"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                health_check = await client.get(f"{ollama_url}/api/tags")
                if health_check.status_code != 200:
                    return {
                        "analysis": None,
                        "ollama_available": False,
                        "error": "Ollama service not responding"
                    }
            except:
                return {
                    "analysis": None,
                    "ollama_available": False,
                    "error": "Ollama service not accessible"
                }
        
        habitability = data.get('habitability_score', {})
        
        prompt = f"""
Analyze this exoplanet as an expert astrobiologist and planetary scientist:

🌍 PLANET: {data.get('name', 'Unknown')}

📊 HABITABILITY ASSESSMENT:
- Overall score: {habitability.get('total_score', 'N/A')}/100
- Survival chance: {habitability.get('survival_chance', 'N/A')}%
- Category: {habitability.get('category', 'Unknown')}

🔬 PHYSICAL PARAMETERS:
{json.dumps(data.get('physical_params', {}), indent=2)}

🛰️ ORBITAL PARAMETERS:
{json.dumps(data.get('orbital_params', {}), indent=2)}

⭐ HOST STAR:
{json.dumps(data.get('host_star', {}), indent=2)}

🔍 DISCOVERY INFORMATION:
{json.dumps(data.get('discovery_info', {}), indent=2)}

⚠️ RISKS:
{json.dumps(habitability.get('risks', []), indent=2)}

Provide a DETAILED analysis:

1. 🏷️ CLASSIFICATION
   - Planet type (rocky/gaseous/icy)
   - Comparison with Solar System planets

2. 🌡️ ENVIRONMENTAL CONDITIONS
   - Temperature regime
   - Gravity and its effects
   - Radiation environment

3. 🧬 HABITABILITY POTENTIAL
   - Possibility of liquid water
   - Atmospheric conditions (if known)
   - Tidal effects

4. 👨‍🚀 COLONIZATION PROSPECTS
   - Required equipment
   - Main challenges
   - Long-term survivability

5. 🔬 SCIENTIFIC SIGNIFICANCE
   - Unique features
   - Value for research

The response should be structured, scientific, yet understandable!
"""

        
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": "gemma2",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "analysis": result.get('response', ''),
                    "ollama_available": True,
                    "error": None
                }
            else:
                return {
                    "analysis": None,
                    "ollama_available": False,
                    "error": f"Ollama returned status {response.status_code}"
                }
                
    except Exception as e:
        logger.error(f"Ollama analysis error: {str(e)}")
        return {
            "analysis": None,
            "ollama_available": False,
            "error": str(e)
        }

# ========== API ENDPOINTS ==========

@api_router.get("/")
async def root():
    return {
        "message": "Enhanced Exoplanet Search API", 
        "version": "2.0",
        "features": [
            "Multi-source data aggregation",
            "Advanced habitability scoring",
            "Real & synthetic image generation",
            "NASA Eyes on Exoplanets integration",
            "AI-powered analysis",
            "Survival chance estimation"
        ]
    }

@api_router.post("/search")
async def search_exoplanet(request: ExoplanetSearchRequest):
    """Enhanced search with habitability scoring and real images"""
    try:
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Aggregate data from all sources
        data = await aggregate_exoplanet_data(request.query, request.search_type)
        
        if not data:
            raise HTTPException(status_code=404, detail="No data found for this exoplanet")
        
        # Calculate habitability score
        habitability = calculate_habitability_score(
            data.get('physical_params', {}),
            data.get('orbital_params', {}),
            data.get('host_star', {})
        )
        data['habitability_score'] = habitability.model_dump()
        
        # Fetch real images
        real_images = await fetch_real_exoplanet_image(
            data['name'], 
            data.get('host_star', {})
        )
        
        # Generate visualizations
        synthetic_star = None
        orbital_diagram = None
        mass_radius_chart = None
        habitability_viz = None
        
        if data.get('host_star'):
            synthetic_star = generate_enhanced_star_image(
                data['host_star'], 
                data['name']
            )
        
        if data.get('orbital_params'):
            orbital_diagram = generate_orbital_diagram(
                data['orbital_params'], 
                data['name']
            )
            
        if data.get('physical_params'):
            mass_radius_chart = generate_mass_radius_chart(
                data['physical_params'], 
                data['name']
            )
        
        # Generate habitability visualization
        habitability_viz = generate_habitability_visualization(
            habitability,
            data['name']
        )
        
        # Compile all visualizations
        data['visualizations'] = {
            'synthetic_star': synthetic_star,
            'orbital_diagram': orbital_diagram,
            'mass_radius_chart': mass_radius_chart,
            'habitability_dashboard': habitability_viz,
            'real_images': real_images.get('real_images', []),
            'nasa_eyes': real_images.get('eyes_on_exoplanets'),
            'hubble_data': real_images.get('hubble_data'),
            'jwst_data': real_images.get('jwst_data')
        }
        
        # Save to database
        exoplanet_obj = ExoplanetData(**data)
        doc = exoplanet_obj.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        await db.exoplanet_searches.insert_one(doc)
        
        logger.info(f"Search completed for {data['name']}: "
                   f"Habitability {habitability.total_score}/100, "
                   f"Survival {habitability.survival_chance}%")
        
        return data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analyze")
async def analyze_exoplanet(request: AIAnalysisRequest):
    """AI analysis with enhanced habitability context"""
    try:
        result = await analyze_with_ollama(request.exoplanet_data, request.ollama_url)
        return result
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {
            "analysis": None,
            "ollama_available": False,
            "error": str(e)
        }

@api_router.get("/habitability/{planet_name}")
async def get_habitability_only(planet_name: str):
    """Get only habitability score for a planet"""
    try:
        data = await aggregate_exoplanet_data(planet_name, "name")
        if not data:
            raise HTTPException(status_code=404, detail="Planet not found")
        
        habitability = calculate_habitability_score(
            data.get('physical_params', {}),
            data.get('orbital_params', {}),
            data.get('host_star', {})
        )
        
        return habitability
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Habitability calculation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/compare")
async def compare_planets(
    planet1: str = Query(..., description="First planet name"),
    planet2: str = Query(..., description="Second planet name")
):
    """Compare two exoplanets"""
    try:
        data1 = await aggregate_exoplanet_data(planet1, "name")
        data2 = await aggregate_exoplanet_data(planet2, "name")
        
        if not data1 or not data2:
            raise HTTPException(status_code=404, detail="One or both planets not found")
        
        hab1 = calculate_habitability_score(
            data1.get('physical_params', {}),
            data1.get('orbital_params', {}),
            data1.get('host_star', {})
        )
        
        hab2 = calculate_habitability_score(
            data2.get('physical_params', {}),
            data2.get('orbital_params', {}),
            data2.get('host_star', {})
        )
        
        return {
            "planet1": {
                "name": planet1,
                "habitability": hab1,
                "physical_params": data1.get('physical_params'),
                "orbital_params": data1.get('orbital_params')
            },
            "planet2": {
                "name": planet2,
                "habitability": hab2,
                "physical_params": data2.get('physical_params'),
                "orbital_params": data2.get('orbital_params')
            },
            "comparison": {
                "better_habitability": planet1 if hab1.total_score > hab2.total_score else planet2,
                "score_difference": abs(hab1.total_score - hab2.total_score),
                "survival_difference": abs(hab1.survival_chance - hab2.survival_chance)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(api_router)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()