"""
INTELLIGENT CRATE DIGGING - STREAMLIT APP
A Hybrid Music Recommendation System For Underground Electronic DJs

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import networkx as nx
import pickle
import re
import random
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from huggingface_hub import snapshot_download


# Optional: fuzzy matching
try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Intelligent Crate Digging",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONFIGURABLE PARAMETERS
# ============================================================

# Paths 
# Download data from dataset repo on startup
DATA_PATH = snapshot_download(
    repo_id="benhasy/intelligent-crate-digging-data",
    repo_type="dataset"
)

BASE_PATH = DATA_PATH
COMBINED_PATH = os.path.join(BASE_PATH, "_combined")

# Method weights for Discogs recommendations
WEIGHT_RULE_BASED = 0.35
WEIGHT_KNOWLEDGE_GRAPH = 0.30
WEIGHT_CLUSTERING = 0.20
WEIGHT_JACCARD = 0.15

# Rule-based parameters
YEAR_TOLERANCE_HIGH = 2
YEAR_TOLERANCE_LOW = 4
ARTIST_EXACT_MATCH = 1.0
ARTIST_CLOSE_MATCH = 0.7
ARTIST_PARTIAL_MATCH = 0.4
ARTIST_LOOSE_MATCH = 0.1
LABEL_MATCH_SCORE = 0.8
YEAR_HIGH_SCORE = 0.6
YEAR_LOW_SCORE = 0.3
COUNTRY_MATCH_SCORE = 0.2
CATNO_PREFIX_SCORE = 0.5

# Knowledge graph parameters
HOP_1_COLLAB_SCORE = 1.0
HOP_1_LABEL_SCORE = 0.8
HOP_2_SCORE = 0.4

# Clustering
SAME_CLUSTER_SCORE = 1.0

# XML to Discogs matching weights
XML_WEIGHT_ARTIST = 0.35
XML_WEIGHT_LABEL = 0.35
XML_WEIGHT_YEAR = 0.15
XML_WEIGHT_GENRE = 0.15


# ============================================================
# KEY CONVERSION (Traditional to Camelot)
# ============================================================

TRADITIONAL_TO_CAMELOT = {
    # Major keys
    'C': '8B', 'C major': '8B', 'Cmaj': '8B',
    'Db': '3B', 'C#': '3B', 'Db major': '3B', 'C# major': '3B', 'Dbmaj': '3B', 'C#maj': '3B',
    'D': '10B', 'D major': '10B', 'Dmaj': '10B',
    'Eb': '5B', 'D#': '5B', 'Eb major': '5B', 'D# major': '5B', 'Ebmaj': '5B', 'D#maj': '5B',
    'E': '12B', 'E major': '12B', 'Emaj': '12B',
    'F': '7B', 'F major': '7B', 'Fmaj': '7B',
    'Gb': '2B', 'F#': '2B', 'Gb major': '2B', 'F# major': '2B', 'Gbmaj': '2B', 'F#maj': '2B',
    'G': '9B', 'G major': '9B', 'Gmaj': '9B',
    'Ab': '4B', 'G#': '4B', 'Ab major': '4B', 'G# major': '4B', 'Abmaj': '4B', 'G#maj': '4B',
    'A': '11B', 'A major': '11B', 'Amaj': '11B',
    'Bb': '6B', 'A#': '6B', 'Bb major': '6B', 'A# major': '6B', 'Bbmaj': '6B', 'A#maj': '6B',
    'B': '1B', 'B major': '1B', 'Bmaj': '1B',
    
    # Minor keys
    'Cm': '5A', 'C minor': '5A', 'Cmin': '5A', 'C min': '5A',
    'C#m': '12A', 'Dbm': '12A', 'C# minor': '12A', 'Db minor': '12A', 'C#min': '12A', 'Dbmin': '12A',
    'Dm': '7A', 'D minor': '7A', 'Dmin': '7A', 'D min': '7A',
    'D#m': '2A', 'Ebm': '2A', 'D# minor': '2A', 'Eb minor': '2A', 'D#min': '2A', 'Ebmin': '2A',
    'Em': '9A', 'E minor': '9A', 'Emin': '9A', 'E min': '9A',
    'Fm': '4A', 'F minor': '4A', 'Fmin': '4A', 'F min': '4A',
    'F#m': '11A', 'Gbm': '11A', 'F# minor': '11A', 'Gb minor': '11A', 'F#min': '11A', 'Gbmin': '11A',
    'Gm': '6A', 'G minor': '6A', 'Gmin': '6A', 'G min': '6A',
    'G#m': '1A', 'Abm': '1A', 'G# minor': '1A', 'Ab minor': '1A', 'G#min': '1A', 'Abmin': '1A',
    'Am': '8A', 'A minor': '8A', 'Amin': '8A', 'A min': '8A',
    'A#m': '3A', 'Bbm': '3A', 'A# minor': '3A', 'Bb minor': '3A', 'A#min': '3A', 'Bbmin': '3A',
    'Bm': '10A', 'B minor': '10A', 'Bmin': '10A', 'B min': '10A',
}

def convert_key_to_camelot(key_str):
    """Convert traditional key notation to Camelot. Returns original if already Camelot or unknown."""
    if not key_str or pd.isna(key_str):
        return None
    
    key_str = str(key_str).strip()
    
    # Check if already Camelot format (1A-12A, 1B-12B)
    if re.match(r'^(1[0-2]|[1-9])[AB]$', key_str):
        return key_str
    
    # Try direct lookup (case-sensitive first)
    if key_str in TRADITIONAL_TO_CAMELOT:
        return TRADITIONAL_TO_CAMELOT[key_str]
    
    # Try case-insensitive lookup
    key_lower = key_str.lower()
    for trad, camelot in TRADITIONAL_TO_CAMELOT.items():
        if trad.lower() == key_lower:
            return camelot
    
    return None


# ============================================================
# STYLE NAME MAPPING
# ============================================================

SINGLE_GENRES = {
    'acid_house': 'Acid House',
    'baltimore_club': 'Baltimore Club',
    'bassline': 'Bassline',
    'bleep': 'Bleep',
    'breakbeat': 'Breakbeat',
    'dnb': 'DNB',
    'donk': 'Donk',
    'dubstep': 'Dubstep',
    'electro': 'Electro',
    'electro_funk': 'Electro Funk',
    'footwork': 'Footwork',
    'freetekno': 'Freetekno',
    'gabber': 'Gabber',
    'ghetto': 'Ghetto',
    'ghetto_house': 'Ghetto House',
    'ghettotech': 'Ghettotech',
    'happy_hardcore': 'Happy Hardcore',
    'italo_house': 'Italo House',
    'juke': 'Juke',
    'jungle': 'Jungle',
    'makina': 'Makina',
    'miami_bass': 'Miami Bass',
    'speed_garage': 'Speed Garage',
    'tribal': 'Tribal',
    'uk_funky': 'UK Funky',
    'uk_garage': 'UK Garage',
    'dj_battle_tool': 'DJ Battle Tool',
}

COMBINED_GENRES = {
    'breakbeat_acid': 'Breakbeat & Acid',
    'breakbeat_happy_hardcore': 'Breakbeat & Happy Hardcore',
    'breakbeat_hardcore_happy_hardcore': 'Breakbeat, Hardcore & Happy Hardcore',
    'breakbeat_hardcore_jungle': 'Breakbeat, Hardcore & Jungle',
    'breakbeat_hardcore_techno': 'Breakbeat, Hardcore & Techno',
    'breakbeat_hardcore': 'Breakbeat & Hardcore',
    'breakbeat_house': 'Breakbeat & House',
    'britcore_breakbeat_hardcore': 'Britcore, Breakbeat & Hardcore',
    'dnb_samba': 'DNB & Samba',
    'dub_jungle': 'Dub & Jungle',
    'hardcore_acid': 'Hardcore & Acid',
    'hardcore_jungle': 'Hardcore & Jungle',
    'hardcore_techno_jungle': 'Hardcore, Techno & Jungle',
    'hiphop_breakbeat': 'Hip-Hop & Breakbeat',
    'jungle_techno': 'Jungle & Techno',
    'techno_deep_acid': 'Techno, Deep & Acid',
    'techno_future_jazz': 'Techno & Future Jazz',
    'tribal_freetekno': 'Tribal & Freetekno',
}

def get_display_name(folder_name):
    """Convert folder name to display name."""
    if folder_name in SINGLE_GENRES:
        return SINGLE_GENRES[folder_name]
    if folder_name in COMBINED_GENRES:
        return COMBINED_GENRES[folder_name]
    return folder_name.replace('_', ' ').title()


def get_folder_name(display_name):
    """Convert display name back to folder name."""
    for folder, display in SINGLE_GENRES.items():
        if display == display_name:
            return folder
    for folder, display in COMBINED_GENRES.items():
        if display == display_name:
            return folder
    return display_name.lower().replace(' ', '_').replace('&', '').replace(',', '')


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_artists(artist_string):
    """Parse multiple artists from artist string."""
    if pd.isna(artist_string) or artist_string == '':
        return []
    
    artist_string = str(artist_string).strip()
    separators = [
        ' feat. ', ' Feat. ', ' ft. ', ' Ft. ',
        ' vs ', ' vs. ', ' Vs ', ' Vs. ',
        ' x ', ' X ',
        ' & ',
        ' and ', ' And ', ' AND ',
        ', ',
        '/', ' / ',
        ','
    ]
    
    artists = [artist_string]
    for sep in separators:
        new_artists = []
        for a in artists:
            new_artists.extend(a.split(sep))
        artists = new_artists
    
    return [a.strip() for a in artists if a.strip() and len(a.strip()) > 1]


def get_catno_prefix(catno):
    """Extract prefix from catalogue number."""
    if pd.isna(catno) or catno == '' or catno == 'N0N3 - 000':
        return None
    match = re.match(r'^([A-Za-z]+)', str(catno).strip())
    return match.group(1).upper() if match else None


# ============================================================
# XML PARSING (REKORDBOX)
# ============================================================

def parse_rekordbox_xml(xml_content):
    """Parse Rekordbox XML and extract track information."""
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        st.error(f"Failed to parse XML: {e}")
        return None
    
    tracks = []
    
    # Find all TRACK elements
    for track in root.iter('TRACK'):
        track_data = {
            'track_id': track.get('TrackID', ''),
            'title': track.get('Name', 'N/A'),
            'artist': track.get('Artist', 'N/A'),
            'album': track.get('Album', ''),
            'genre': track.get('Genre', ''),
            'bpm': track.get('AverageBpm', ''),
            'key_original': track.get('Tonality', ''),
            'year': track.get('Year', ''),
            'label': track.get('Label', ''),
            'remixer': track.get('Remixer', ''),
            'comments': track.get('Comments', ''),
        }
        
        # Convert BPM to float
        try:
            track_data['bpm'] = float(track_data['bpm']) if track_data['bpm'] else None
        except ValueError:
            track_data['bpm'] = None
        
        # Convert year to int
        try:
            track_data['year'] = int(track_data['year']) if track_data['year'] else None
        except ValueError:
            track_data['year'] = None
        
        # Convert key to Camelot (store both original and converted)
        original_key = track_data['key_original']
        camelot_key = convert_key_to_camelot(original_key)
        track_data['key_camelot'] = camelot_key
        
        # Only add tracks with at least a title or artist
        if track_data['title'] != 'N/A' or track_data['artist'] != 'N/A':
            tracks.append(track_data)
    
    if not tracks:
        return None
    
    return pd.DataFrame(tracks)


# ============================================================
# BPM & KEY COMPATIBILITY (FOR XML LIBRARY)
# ============================================================

def check_bpm_compatible(bpm1, bpm2, tolerance=0.08):
    """Check BPM compatibility including half-time and double-time."""
    if bpm1 is None or bpm2 is None:
        return False, 0, None
    
    # Check direct BPM match
    lower = bpm1 * (1 - tolerance) 
    upper = bpm1 * (1 + tolerance)
    if lower <= bpm2 <= upper:
        return True, bpm2 - bpm1, "direct"
    
    # Check half-time (bpm2 is half of bpm1)
    half_bpm1 = bpm1 / 2
    lower = half_bpm1 * (1 - tolerance)
    upper = half_bpm1 * (1 + tolerance)
    if lower <= bpm2 <= upper:
        return True, bpm2 - half_bpm1, "half-time"
    
    # Check double-time (bpm2 is double bpm1)
    double_bpm1 = bpm1 * 2
    lower = double_bpm1 * (1 - tolerance)
    upper = double_bpm1 * (1 + tolerance)
    if lower <= bpm2 <= upper:
        return True, bpm2 - double_bpm1, "double-time"
    
    # Check if bpm1 is half of bpm2
    double_bpm2 = bpm2 * 2
    lower = bpm1 * (1 - tolerance)
    upper = bpm1 * (1 + tolerance)
    if lower <= double_bpm2 <= upper:
        return True, bpm2 - (bpm1 / 2), "half-time"
    
    return False, 0, None


def check_key_compatible(key1, key2):
    """Check key compatibility using Camelot wheel rules."""
    if key1 is None or key2 is None:
        return False
    
    if key1 == key2:
        return True
    
    try:
        num1 = int(key1[:-1])
        letter1 = key1[-1]
        num2 = int(key2[:-1]) 
        letter2 = key2[-1]
    except (ValueError, IndexError):
        return False

    # Same number, different letter (relative major/minor)
    if num1 == num2 and letter1 != letter2:
        return True
    
    # Adjacent numbers, same letter (with wheel wrap)
    if letter1 == letter2:
        diff = abs(num1 - num2)
        if diff == 1 or diff == 11:
            return True
    
    return False


def find_mix_compatible_tracks(tracks_df, seed_idx):
    """Find tracks compatible for mixing (BPM + Key) from user's library."""
    seed_track = tracks_df.iloc[seed_idx]
    seed_bpm = seed_track['bpm']
    seed_key = seed_track['key_camelot']
    
    compatible_tracks = []
    seen_tracks = set()
    
    for idx, row in tracks_df.iterrows():
        if idx == seed_idx:
            continue
        
        # Skip duplicates
        track_key = f"{row['title']}_{row['artist']}".lower().strip()
        if track_key in seen_tracks:
            continue
        seen_tracks.add(track_key)
        
        bpm_ok, bpm_diff, bpm_type = check_bpm_compatible(seed_bpm, row['bpm'])
        key_ok = check_key_compatible(seed_key, row['key_camelot'])
        
        if bpm_ok and key_ok:
            bpm_diff_pct = abs(bpm_diff) / seed_bpm if seed_bpm else 0
            bpm_score = 1 - bpm_diff_pct
            if bpm_type != "direct":
                bpm_score *= 0.9
            key_score = 1.0 if seed_key == row['key_camelot'] else 0.8
            combined_score = (bpm_score * 0.3) + (key_score * 0.7)
            
            # Format reason string
            if bpm_type == "direct":
                bpm_reason = f"BPM: {row['bpm']:.1f} ({'+' if bpm_diff >= 0 else ''}{int(bpm_diff)})"
            else:
                bpm_reason = f"BPM: {row['bpm']:.1f} ({bpm_type})"
            
            # Display key in original format if available
            display_key = row['key_original'] if row['key_original'] else row['key_camelot']
            
            compatible_tracks.append({
                'idx': idx,
                'title': row['title'],
                'artist': row['artist'],
                'bpm': row['bpm'],
                'key': display_key,
                'key_camelot': row['key_camelot'],
                'score': round(combined_score, 3),
                'reason': f"{bpm_reason}, Key: {display_key}"
            })
    
    compatible_tracks.sort(key=lambda x: x['score'], reverse=True)
    return compatible_tracks


# ============================================================
# XML TO DISCOGS MATCHING
# ============================================================

def find_discogs_matches(xml_track, discogs_df, n=20):
    """Find matching Discogs releases for an XML track using weighted scoring."""
    
    # Extract available metadata from XML track
    xml_artist = str(xml_track.get('artist', '')).strip() if pd.notna(xml_track.get('artist')) else ''
    xml_label = str(xml_track.get('label', '')).strip() if pd.notna(xml_track.get('label')) else ''
    xml_year = xml_track.get('year') if pd.notna(xml_track.get('year')) else None
    xml_genre = str(xml_track.get('genre', '')).strip() if pd.notna(xml_track.get('genre')) else ''
    
    # Determine which fields are available and calculate weight normalisation
    available_weights = {}
    if xml_artist:
        available_weights['artist'] = XML_WEIGHT_ARTIST
    if xml_label:
        available_weights['label'] = XML_WEIGHT_LABEL
    if xml_year:
        available_weights['year'] = XML_WEIGHT_YEAR
    if xml_genre:
        available_weights['genre'] = XML_WEIGHT_GENRE
    
    if not available_weights:
        return [], []
    
    # Normalise weights
    total_weight = sum(available_weights.values())
    normalised_weights = {k: v / total_weight for k, v in available_weights.items()}
    
    candidates = []
    
    for idx, row in discogs_df.iterrows():
        score = 0
        reasons = []
        
        # Artist matching
        if 'artist' in normalised_weights:
            discogs_artist = str(row.get('parsed_artist', '')).lower() if pd.notna(row.get('parsed_artist')) else ''
            xml_artist_lower = xml_artist.lower()
            
            if xml_artist_lower and discogs_artist:
                if xml_artist_lower == discogs_artist:
                    score += normalised_weights['artist'] * 1.0
                    reasons.append("exact artist match")
                elif xml_artist_lower in discogs_artist or discogs_artist in xml_artist_lower:
                    score += normalised_weights['artist'] * 0.7
                    reasons.append("partial artist match")
                elif FUZZY_AVAILABLE:
                    ratio = fuzz.ratio(xml_artist_lower, discogs_artist)
                    if ratio >= 80:
                        score += normalised_weights['artist'] * (ratio / 100)
                        reasons.append(f"similar artist ({ratio}%)")
        
        # Label matching
        if 'label' in normalised_weights:
            discogs_label = str(row.get('label', '')).lower() if pd.notna(row.get('label')) else ''
            xml_label_lower = xml_label.lower()
            
            if xml_label_lower and discogs_label and 'not on label' not in discogs_label:
                if xml_label_lower in discogs_label or discogs_label in xml_label_lower:
                    score += normalised_weights['label'] * 1.0
                    reasons.append(f"same label")
        
        # Year matching
        if 'year' in normalised_weights and xml_year:
            discogs_year = row.get('oldest_year') if pd.notna(row.get('oldest_year')) else row.get('year')
            if discogs_year and discogs_year > 0:
                try:
                    year_diff = abs(int(xml_year) - int(discogs_year))
                    if year_diff <= 2:
                        score += normalised_weights['year'] * 1.0
                        reasons.append(f"same era ({int(discogs_year)})")
                    elif year_diff <= 5:
                        score += normalised_weights['year'] * 0.5
                        reasons.append(f"similar era ({int(discogs_year)})")
                except (ValueError, TypeError):
                    pass
        
        # Genre/style matching
        if 'genre' in normalised_weights:
            discogs_style = str(row.get('style', '')).lower() if pd.notna(row.get('style')) else ''
            xml_genre_lower = xml_genre.lower()
            
            if xml_genre_lower and discogs_style:
                xml_genres = set(g.strip() for g in xml_genre_lower.split(','))
                discogs_styles = set(s.strip() for s in discogs_style.split(','))
                
                intersection = xml_genres & discogs_styles
                if intersection:
                    score += normalised_weights['genre'] * 1.0
                    reasons.append(f"style match")
                else:
                    # Partial matching
                    for xg in xml_genres:
                        for ds in discogs_styles:
                            if xg in ds or ds in xg:
                                score += normalised_weights['genre'] * 0.5
                                reasons.append(f"partial style match")
                                break
                        else:
                            continue
                        break
        
        if score > 0:
            candidates.append({
                'idx': idx,
                'row': row,
                'score': score,
                'reasons': reasons
            })
    
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Split into top matches and discovery (same label, low similarity)
    top_matches = candidates[:n]
    
    # Discovery: same label but lower similarity
    discovery = []
    if xml_label:
        xml_label_lower = xml_label.lower()
        same_label_candidates = []
        
        for c in candidates:
            discogs_label = str(c['row'].get('label', '')).lower() if pd.notna(c['row'].get('label')) else ''
            if xml_label_lower in discogs_label or discogs_label in xml_label_lower:
                if c not in top_matches[:5]:  # Exclude top 5
                    same_label_candidates.append(c)
        
        # Get low similarity ones for discovery
        for threshold in [0.15, 0.25, 0.35, 0.40]:
            discovery_pool = [c for c in same_label_candidates if c['score'] <= threshold]
            if len(discovery_pool) >= 5:
                discovery = random.sample(discovery_pool, 5)
                break
        else:
            if same_label_candidates:
                discovery = random.sample(same_label_candidates, min(5, len(same_label_candidates)))
    
    return top_matches, discovery


# ============================================================
# DATA LOADING (CACHED)
# ============================================================

@st.cache_data
def get_available_formats():
    """Get list of available format CSVs."""
    formats = []
    if os.path.exists(COMBINED_PATH):
        for f in os.listdir(COMBINED_PATH):
            if f.startswith('all_') and f.endswith('_processed.csv'):
                fmt = f.replace('all_', '').replace('_processed.csv', '')
                formats.append(fmt)
    return sorted(formats)


@st.cache_data
def get_available_styles():
    """Get list of available style folders."""
    styles = []
    if os.path.exists(BASE_PATH):
        for d in os.listdir(BASE_PATH):
            if os.path.isdir(os.path.join(BASE_PATH, d)) and d != '_combined':
                styles.append(d)
    return sorted(styles)


@st.cache_data
def load_combined_csv(format_name):
    """Load a combined format CSV."""
    path = os.path.join(COMBINED_PATH, f"all_{format_name}_processed.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'artists_list' not in df.columns:
            df['artists_list'] = df['parsed_artist'].apply(parse_artists)
        else:
            df['artists_list'] = df['artists_list'].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else []
            )
        if 'oldest_year' not in df.columns:
            df['oldest_year'] = df['year']
        return df
    return None


@st.cache_data
def load_style_format_csv(style, format_name):
    """Load a specific style/format CSV."""
    path = os.path.join(BASE_PATH, style, f"processed_{format_name}", 
                        f"{style}_{format_name}_processed.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'artists_list' not in df.columns:
            df['artists_list'] = df['parsed_artist'].apply(parse_artists)
        else:
            df['artists_list'] = df['artists_list'].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else []
            )
        if 'oldest_year' not in df.columns:
            df['oldest_year'] = df['year']
        return df
    return None


def load_graph(style, format_name):
    """Load knowledge graph for a style/format."""
    path = os.path.join(BASE_PATH, style, f"processed_{format_name}",
                        f"{style}_{format_name}_graph.pkl")
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


# ============================================================
# RECOMMENDATION ENGINE (FOR DISCOGS DATA)
# ============================================================

class RecommendationEngine:
    def __init__(self, df, graph=None):
        self.df = df
        self.graph = graph
    
    def _artist_similarity(self, artists1, artists2):
        if not artists1 or not artists2:
            return 0, []
        
        best_score = 0
        for a1 in artists1:
            for a2 in artists2:
                a1_lower, a2_lower = a1.lower().strip(), a2.lower().strip()
                
                if a1_lower == a2_lower:
                    score = ARTIST_EXACT_MATCH
                elif FUZZY_AVAILABLE:
                    ratio = fuzz.ratio(a1_lower, a2_lower)
                    if ratio >= 90:
                        score = ARTIST_CLOSE_MATCH
                    elif ratio >= 70:
                        score = ARTIST_PARTIAL_MATCH
                    elif ratio >= 50:
                        score = ARTIST_LOOSE_MATCH
                    else:
                        score = 0
                else:
                    score = ARTIST_PARTIAL_MATCH if (a1_lower in a2_lower or a2_lower in a1_lower) else 0
                
                if score > best_score:
                    best_score = score
        
        return best_score, []
    
    def _label_similarity(self, labels1, labels2):
        if pd.isna(labels1) or pd.isna(labels2):
            return 0, []
        
        set1 = set(l.strip().lower() for l in str(labels1).split(',') if 'not on label' not in l.lower())
        set2 = set(l.strip().lower() for l in str(labels2).split(',') if 'not on label' not in l.lower())
        
        matches = set1 & set2
        return (LABEL_MATCH_SCORE, list(matches)) if matches else (0, [])
    
    def _year_similarity(self, year1, year2):
        if pd.isna(year1) or pd.isna(year2) or year1 == 0 or year2 == 0:
            return 0
        
        diff = abs(int(year1) - int(year2))
        if diff <= YEAR_TOLERANCE_HIGH:
            return YEAR_HIGH_SCORE
        elif diff <= YEAR_TOLERANCE_LOW:
            return YEAR_LOW_SCORE
        return 0
    
    def _catno_similarity(self, catno1, catno2):
        prefix1, prefix2 = get_catno_prefix(catno1), get_catno_prefix(catno2)
        return CATNO_PREFIX_SCORE if (prefix1 and prefix2 and prefix1 == prefix2) else 0
    
    def _rule_based_score(self, seed_row, candidate_row):
        score = 0
        reasons = []
        
        seed_artists = seed_row['artists_list'] if isinstance(seed_row['artists_list'], list) else []
        cand_artists = candidate_row['artists_list'] if isinstance(candidate_row['artists_list'], list) else []
        
        artist_score, _ = self._artist_similarity(seed_artists, cand_artists)
        if artist_score > 0:
            score += artist_score
            reasons.append("same/similar artist" if artist_score == ARTIST_EXACT_MATCH else "related artist")
        
        label_score, label_matches = self._label_similarity(seed_row['label'], candidate_row['label'])
        if label_score > 0:
            score += label_score
            label_name = label_matches[0][:25] + '...' if len(label_matches[0]) > 25 else label_matches[0]
            reasons.append(f"same label ({label_name})")
        
        year_score = self._year_similarity(seed_row.get('oldest_year', seed_row['year']),
                                           candidate_row.get('oldest_year', candidate_row['year']))
        if year_score > 0:
            score += year_score
            reasons.append(f"similar era ({int(candidate_row.get('oldest_year', candidate_row['year']))})")
        
        if pd.notna(seed_row['country']) and pd.notna(candidate_row['country']):
            if seed_row['country'] == candidate_row['country']:
                score += COUNTRY_MATCH_SCORE
                reasons.append(f"same country")
        
        catno_score = self._catno_similarity(
            seed_row.get('catno_cleaned', seed_row.get('catno', '')),
            candidate_row.get('catno_cleaned', candidate_row.get('catno', ''))
        )
        if catno_score > 0:
            score += catno_score
            reasons.append("same catalogue series")
        
        max_possible = ARTIST_EXACT_MATCH + LABEL_MATCH_SCORE + YEAR_HIGH_SCORE + COUNTRY_MATCH_SCORE + CATNO_PREFIX_SCORE
        return score / max_possible, reasons
    
    def _knowledge_graph_score(self, seed_row, candidate_row):
        if self.graph is None:
            return 0, []
        
        seed_artists = seed_row['artists_list'] if isinstance(seed_row['artists_list'], list) else []
        cand_artists = candidate_row['artists_list'] if isinstance(candidate_row['artists_list'], list) else []
        
        if not seed_artists or not cand_artists:
            return 0, []
        
        # Skip non-artist entries
        skip_artists = {'various', 'various artists', 'unknown', 'unknown artist', 'no artist'}
        
        best_score = 0
        best_reason = []
        
        for s_artist in seed_artists:
            if s_artist.lower() in skip_artists:
                continue
            if s_artist not in self.graph:
                continue
            for c_artist in cand_artists:
                if c_artist.lower() in skip_artists:
                    continue
                if c_artist not in self.graph or s_artist.lower() == c_artist.lower():
                    continue
                
                if self.graph.has_edge(s_artist, c_artist):
                    relationship = self.graph[s_artist][c_artist].get('relationship', 'connected')
                    if relationship == 'collaborated_with' and HOP_1_COLLAB_SCORE > best_score:
                        best_score = HOP_1_COLLAB_SCORE
                        best_reason = ["artists collaborated"]
                    elif relationship == 'same_label' and HOP_1_LABEL_SCORE > best_score:
                        best_score = HOP_1_LABEL_SCORE
                        best_reason = ["same label network"]
                else:
                    try:
                        path = nx.shortest_path(self.graph, s_artist, c_artist)
                        if len(path) == 3 and HOP_2_SCORE > best_score:
                            # Skip if middle node is Various
                            if path[1].lower() not in skip_artists:
                                best_score = HOP_2_SCORE
                                best_reason = [f"connected via {path[1][:15]}..."]
                    except nx.NetworkXNoPath:
                        pass
        
        return best_score, best_reason
    
    def _clustering_score(self, seed_row, candidate_row):
        if 'cluster' not in seed_row or 'cluster' not in candidate_row:
            return 0, []
        if pd.isna(seed_row['cluster']) or pd.isna(candidate_row['cluster']):
            return 0, []
        if int(seed_row['cluster']) == int(candidate_row['cluster']):
            return SAME_CLUSTER_SCORE, [f"same cluster"]
        return 0, []
    
    def _jaccard_score(self, seed_row, candidate_row):
        seed_styles = set(s.strip().lower() for s in str(seed_row.get('style', '')).split(',')) if pd.notna(seed_row.get('style')) else set()
        cand_styles = set(s.strip().lower() for s in str(candidate_row.get('style', '')).split(',')) if pd.notna(candidate_row.get('style')) else set()
        
        if not seed_styles or not cand_styles:
            return 0, []
        
        intersection = seed_styles & cand_styles
        union = seed_styles | cand_styles
        jaccard = len(intersection) / len(union) if union else 0
        
        return (jaccard, [f"style match ({jaccard:.0%})"]) if jaccard > 0 else (0, [])
    
    def get_recommendations(self, seed_idx, n=50, exclude_same_artist=False):
        seed_row = self.df.iloc[seed_idx]
        seed_actual_idx = self.df.index[seed_idx]
        seed_artists = set(a.lower() for a in (seed_row['artists_list'] if isinstance(seed_row['artists_list'], list) else []))
        
        # Get seed labels for discovery matching
        seed_labels = set()
        if pd.notna(seed_row['label']):
            seed_labels = set(l.strip().lower() for l in str(seed_row['label']).split(',') if 'not on label' not in l.lower())
        
        candidates = []
        
        for idx, candidate_row in self.df.iterrows():
            if idx == seed_actual_idx:
                continue
            
            if exclude_same_artist:
                cand_artists = set(a.lower() for a in (candidate_row['artists_list'] if isinstance(candidate_row['artists_list'], list) else []))
                if seed_artists & cand_artists:
                    continue
            
            scores = {}
            all_reasons = []
            
            scores['rule_based'], reasons = self._rule_based_score(seed_row, candidate_row)
            all_reasons.extend(reasons)
            
            scores['knowledge_graph'], reasons = self._knowledge_graph_score(seed_row, candidate_row)
            all_reasons.extend(reasons)
            
            scores['clustering'], reasons = self._clustering_score(seed_row, candidate_row)
            all_reasons.extend(reasons)
            
            scores['jaccard'], reasons = self._jaccard_score(seed_row, candidate_row)
            all_reasons.extend(reasons)
            
            final_score = (
                scores['rule_based'] * WEIGHT_RULE_BASED +
                scores['knowledge_graph'] * WEIGHT_KNOWLEDGE_GRAPH +
                scores['clustering'] * WEIGHT_CLUSTERING +
                scores['jaccard'] * WEIGHT_JACCARD
            )
            
            # Check if same label for discovery
            cand_labels = set()
            if pd.notna(candidate_row['label']):
                cand_labels = set(l.strip().lower() for l in str(candidate_row['label']).split(',') if 'not on label' not in l.lower())
            same_label = bool(seed_labels & cand_labels)
            
            if final_score > 0:
                candidates.append({
                    'idx': idx,
                    'row': candidate_row,
                    'score': final_score,
                    'breakdown': scores,
                    'reasons': all_reasons,
                    'same_label': same_label
                })
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Top 5 stay as best matches
        top_5 = candidates[:5]
        top_5_indices = {c['idx'] for c in top_5}
        
        # Discovery picks: combine two strategies
        discovery = []
        
        # Strategy 1: Same label, different vibe (low similarity)
        same_label_candidates = [c for c in candidates if c['same_label'] and c['idx'] not in top_5_indices]
        label_discoveries = []
        
        for threshold in [0.15, 0.25, 0.35, 0.40]:
            discovery_pool = [c for c in same_label_candidates if c['score'] <= threshold]
            if len(discovery_pool) >= 3:
                label_discoveries = random.sample(discovery_pool, min(3, len(discovery_pool)))
                break
        else:
            if len(same_label_candidates) > 0:
                label_discoveries = random.sample(same_label_candidates, min(3, len(same_label_candidates)))
        
        # Add discovery reason
        for d in label_discoveries:
            d['discovery_reason'] = 'same label, different style'
        
        # Strategy 2: 2-hop graph connections (if graph available)
        graph_discoveries = []
        
        # Skip non-artist entries
        skip_artists = {'various', 'various artists', 'unknown', 'unknown artist', 'no artist'}
        
        if self.graph is not None:
            # Find candidates with 2-hop connections that aren't already in top 5 or label discoveries
            label_discovery_indices = {d['idx'] for d in label_discoveries}
            two_hop_candidates = []
            
            # Use ORIGINAL case artists for graph lookup (graph stores original case)
            seed_artists_original = seed_row['artists_list'] if isinstance(seed_row['artists_list'], list) else []
            
            for c in candidates:
                if c['idx'] in top_5_indices or c['idx'] in label_discovery_indices:
                    continue
                
                cand_artists = c['row']['artists_list'] if isinstance(c['row']['artists_list'], list) else []
                
                found_path = False
                for s_artist in seed_artists_original:
                    if s_artist.lower() in skip_artists:
                        continue
                    if s_artist not in self.graph or found_path:
                        continue
                    
                    for c_artist in cand_artists:
                        if c_artist.lower() in skip_artists:
                            continue
                        if c_artist not in self.graph:
                            continue
                        
                        # Check if 2-hop connection exists
                        try:
                            path = nx.shortest_path(self.graph, s_artist, c_artist)
                            if len(path) == 3:  # Exactly 2 hops
                                middle_artist = path[1]
                                # Skip if middle node is Various
                                if middle_artist.lower() not in skip_artists:
                                    c['discovery_reason'] = f"connected via {middle_artist[:20]}"
                                    two_hop_candidates.append(c)
                                    found_path = True
                                    break
                        except nx.NetworkXNoPath:
                            continue
            
            # Pick 2 random from 2-hop candidates (prefer lower similarity for variety)
            if two_hop_candidates:
                two_hop_candidates.sort(key=lambda x: x['score'])  # Sort by score (low to high)
                graph_discoveries = random.sample(two_hop_candidates[:10], min(2, len(two_hop_candidates)))
        
        # Combine both strategies
        discovery = label_discoveries + graph_discoveries
        
        # If we don't have 5 discoveries yet, fill with remaining same_label candidates
        if len(discovery) < 5 and same_label_candidates:
            remaining = [c for c in same_label_candidates if c not in discovery]
            if remaining:
                needed = 5 - len(discovery)
                additional = random.sample(remaining, min(needed, len(remaining)))
                for d in additional:
                    d['discovery_reason'] = 'same label, different style'
                discovery.extend(additional)
        
        for d in discovery:
            d['is_discovery'] = True
        
        discovery_indices = {c['idx'] for c in discovery}
        
        used_indices = top_5_indices | discovery_indices
        rest = [c for c in candidates[5:] if c['idx'] not in used_indices]
        random.shuffle(rest)
        
        return {
            'top_5': top_5,
            'discovery': discovery,
            'rest': rest[:n-10] if len(rest) > n-10 else rest
        }


# ============================================================
# STREAMLIT UI
# ============================================================

def main():
    st.title("🪐 Intelligent Crate Digging 🪐")
    st.markdown("*A Hybrid Music Recommendation System For Underground Electronic DJs. Enjoy! - Hasy*")
    
    # Sidebar
    st.sidebar.header("🔧 Settings")
    
    # Format selection
    formats_raw = get_available_formats()
    if not formats_raw:
        st.error(f"No data found in {COMBINED_PATH}. Please check the BASE_PATH.")
        return
    
    def format_display_name(fmt):
        if fmt.lower() == 'cd':
            return 'CD'
        elif fmt.lower() == 'vhs':
            return 'VHS'
        else:
            return fmt.capitalize()
    
    format_options = ['All'] + [format_display_name(fmt) for fmt in formats_raw if fmt.lower() != 'formats']
    
    selected_format_display = st.sidebar.selectbox("Select Format", format_options)
    
    if selected_format_display == 'All':
        selected_format = 'formats'
    else:
        selected_format = selected_format_display.lower()
    
    # Load data
    with st.spinner(f"Loading {selected_format_display} data..."):
        df = load_combined_csv(selected_format)
    
    if df is None:
        st.error(f"Could not load data for {selected_format}")
        return
    
    st.sidebar.success(f"Loaded {len(df):,} releases")
    
    # Filters
    st.sidebar.header("🎚️ Filters")
    
    available_styles = get_available_styles()
    style_display_names = ['All'] + sorted([get_display_name(s) for s in available_styles])
    selected_primary_style = st.sidebar.selectbox("Primary Style", style_display_names)
    
    primary_filtered_df = df.copy()
    if selected_primary_style != 'All':
        folder_name = get_folder_name(selected_primary_style)
        if '_source_style' in primary_filtered_df.columns:
            primary_filtered_df = primary_filtered_df[primary_filtered_df['_source_style'] == folder_name]
    
    all_style_tags = set()
    for styles in primary_filtered_df['style'].dropna():
        for s in str(styles).split(','):
            tag = s.strip()
            if tag:
                all_style_tags.add(tag)
    
    additional_style_options = sorted(all_style_tags)
    selected_additional_styles = st.sidebar.multiselect(
        "Additional Style Tags (optional)", 
        additional_style_options,
        help="Filter by specific style tags within the selected primary style"
    )
    
    valid_years = df['oldest_year'].dropna()
    valid_years = valid_years[valid_years > 0]
    
    if len(valid_years) > 0:
        min_year = int(valid_years.min())
        max_year = int(valid_years.max())
    else:
        min_year, max_year = 1990, 2025
    
    year_range = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))
    
    countries = ['All'] + sorted(df['country'].dropna().unique().tolist())
    selected_country = st.sidebar.selectbox("Country", countries)
    
    # Apply filters
    filtered_df = primary_filtered_df.copy()
    
    if selected_additional_styles:
        for style_tag in selected_additional_styles:
            filtered_df = filtered_df[filtered_df['style'].str.contains(style_tag, case=False, na=False)]
    
    filtered_df = filtered_df[(filtered_df['oldest_year'] >= year_range[0]) & 
                               (filtered_df['oldest_year'] <= year_range[1])]
    
    if selected_country != 'All':
        filtered_df = filtered_df[filtered_df['country'] == selected_country]
    
    st.sidebar.info(f"Showing {len(filtered_df):,} releases after filters")
    
    exclude_same_artist = st.sidebar.checkbox("Exclude same artist from recommendations", value=False)
    
    # Disclaimer for My Library tab
    st.sidebar.markdown("---")
    st.sidebar.caption("⚠️ Note: Sidebar filters apply to Discogs data (Tabs 1 & 2). Excessive filtering may limit results for 'My Library' Discogs recommendations.")
    
    # Main content - TABS
    tab1, tab2, tab3 = st.tabs(["🔍 Search & Recommend", "📋 Browse All", "📁 My Library"])
    
    # ========================================
    # TAB 1: SEARCH & RECOMMEND (DISCOGS)
    # ========================================
    with tab1:
        st.header("Search for a Release")
        
        if len(filtered_df) == 0:
            st.warning("No releases match your current filters. Try adjusting the filters in the sidebar.")
        else:
            search_query = st.text_input("Search by artist or title", placeholder="e.g., Manix, Galaxy 2 Galaxy, Noise Factory....", key="tab1_search")
            
            if search_query:
                query_lower = search_query.lower().strip()
                results = filtered_df[
                    filtered_df['title'].str.lower().str.contains(query_lower, na=False) |
                    filtered_df['parsed_artist'].str.lower().str.contains(query_lower, na=False)
                ]
                
                if len(results) == 0:
                    st.warning("No results found. Try a different search term.")
                else:
                    st.success(f"Found {len(results)} releases")
                    
                    for idx, row in results.head(20).iterrows():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            year = int(row.get('oldest_year', row['year'])) if pd.notna(row.get('oldest_year', row['year'])) else 'Unknown'
                            st.markdown(f"**{row['title']}** ({year})")
                            label_display = str(row['label'])[:60] + '...' if pd.notna(row['label']) and len(str(row['label'])) > 60 else row['label']
                            st.caption(f"Label: {label_display}")
                        with col2:
                            if st.button("Get Recs", key=f"rec_{idx}"):
                                st.session_state['selected_release_idx'] = idx
                                st.session_state['selected_release_title'] = row['title']
            
            if 'selected_release_idx' in st.session_state:
                st.divider()
                st.header(f"Recommendations for: {st.session_state['selected_release_title']}")
                
                selected_idx = st.session_state['selected_release_idx']
                
                # Load knowledge graph if specific style selected
                graph_to_use = None

                if selected_primary_style != 'All':
                    folder_name = get_folder_name(selected_primary_style)
                    
                    # Determine which graph to load based on format selection
                    if selected_format == 'formats':
                        # User selected "All formats" - use combined graph at style root
                        graph_file = f"{folder_name}_all_formats_graph.pkl"
                        graph_path = os.path.join(BASE_PATH, folder_name, graph_file)
                        format_display = "all formats"
                    else:
                        # User selected specific format - look in processed subfolder
                        graph_file = f"{folder_name}_{selected_format}_graph.pkl"
                        graph_path = os.path.join(BASE_PATH, folder_name, f"processed_{selected_format}", graph_file)
                        format_display = selected_format
                    
                    with st.spinner("Loading knowledge graph..."):
                        if os.path.exists(graph_path):
                            with open(graph_path, 'rb') as f:
                                graph_to_use = pickle.load(f)
                    
                    if graph_to_use:
                        st.success(f"✅ Using knowledge graph for {selected_primary_style} {format_display}")
                        st.caption(f"Graph: {graph_to_use.number_of_nodes():,} nodes, {graph_to_use.number_of_edges():,} edges")
                    else:
                        st.warning(f"⚠️ No knowledge graph found for {selected_primary_style} {format_display}")
                else:
                    st.info("ℹ️ Knowledge graph disabled - select a specific style to enable")

                engine = RecommendationEngine(filtered_df, graph=graph_to_use)
                
                if selected_idx in filtered_df.index:
                    position = filtered_df.index.get_loc(selected_idx)
                    
                    with st.spinner("Finding recommendations..."):
                        results = engine.get_recommendations(position, n=50, exclude_same_artist=exclude_same_artist)
                    
                    if results['top_5'] or results['discovery'] or results['rest']:
                        total_recs = len(results['top_5']) + len(results['discovery']) + len(results['rest'])
                        st.success(f"Found {total_recs} recommendations")
                        
                        if results['top_5']:
                            st.subheader("⭐ Top 5 Matches")
                            for i, rec in enumerate(results['top_5'], 1):
                                row = rec['row']
                                score = rec['score']
                                reasons = rec['reasons']
                                
                                with st.container():
                                    col1, col2, col3 = st.columns([3, 1, 1])
                                    
                                    with col1:
                                        year = int(row.get('oldest_year', row['year'])) if pd.notna(row.get('oldest_year', row['year'])) else 'Unknown'
                                        st.markdown(f"### {i}. {row['title']} ({year})")
                                        label_display = str(row['label'])[:50] + '...' if pd.notna(row['label']) and len(str(row['label'])) > 50 else row['label']
                                        st.caption(f"Label: {label_display}")
                                        if reasons:
                                            st.caption(f"💡 {', '.join(reasons[:5])}")
                                    
                                    with col2:
                                        st.metric("Score", f"{score:.2f}")
                                    
                                    with col3:
                                        if pd.notna(row.get('resource_url')):
                                            st.link_button("View on Discogs", row['resource_url'])
                                    
                                    st.divider()
                        
                        if results['discovery']:
                            st.subheader("🔮 Discovery Picks")
                            st.caption("Hidden connections: same-label deep cuts + artists linked through collaborations")
                            for i, rec in enumerate(results['discovery'], 1):
                                row = rec['row']
                                score = rec['score']
                                reasons = rec['reasons']
                                
                                with st.container():
                                    col1, col2, col3 = st.columns([3, 1, 1])
                                    
                                    with col1:
                                        year = int(row.get('oldest_year', row['year'])) if pd.notna(row.get('oldest_year', row['year'])) else 'Unknown'
                                        st.markdown(f"**{i}. {row['title']}** ({year})")
                                        label_display = str(row['label'])[:50] + '...' if pd.notna(row['label']) and len(str(row['label'])) > 50 else row['label']
                                        st.caption(f"Label: {label_display}")
                                        if reasons:
                                            st.caption(f"💡 {', '.join(reasons[:5])}")
                                        # Show discovery reason
                                        if rec.get('discovery_reason'):
                                            st.caption(f"🔮 Discovery: {rec['discovery_reason']}")
                                    
                                    with col2:
                                        st.metric("Score", f"{score:.2f}")
                                    
                                    with col3:
                                        if pd.notna(row.get('resource_url')):
                                            st.link_button("View on Discogs", row['resource_url'])
                                    
                                    st.divider()
                        
                        if results['rest']:
                            st.subheader("🎲 More Recommendations")
                            n_display = st.slider("Show more results", 5, 40, 15, key="tab1_slider")
                            
                            for i, rec in enumerate(results['rest'][:n_display], 1):
                                row = rec['row']
                                score = rec['score']
                                reasons = rec['reasons']
                                
                                with st.container():
                                    col1, col2, col3 = st.columns([3, 1, 1])
                                    
                                    with col1:
                                        year = int(row.get('oldest_year', row['year'])) if pd.notna(row.get('oldest_year', row['year'])) else 'Unknown'
                                        st.markdown(f"**{i}. {row['title']}** ({year})")
                                        label_display = str(row['label'])[:50] + '...' if pd.notna(row['label']) and len(str(row['label'])) > 50 else row['label']
                                        st.caption(f"Label: {label_display}")
                                        if reasons:
                                            st.caption(f"💡 {', '.join(reasons[:5])}")
                                    
                                    with col2:
                                        st.metric("Score", f"{score:.2f}")
                                    
                                    with col3:
                                        if pd.notna(row.get('resource_url')):
                                            st.link_button("View on Discogs", row['resource_url'])
                                    
                                    st.divider()
                    else:
                        st.warning("No recommendations found for this release.")
                else:
                    st.warning("Selected release no longer matches current filters. Please search again.")
    
    # ========================================
    # TAB 2: BROWSE ALL (DISCOGS)
    # ========================================
    with tab2:
        st.header("Browse Database")
        
        if len(filtered_df) == 0:
            st.warning("No releases match your current filters. Try adjusting the filters in the sidebar.")
        else:
            label_search = st.text_input("🔍 Filter by label", placeholder="e.g., Reinforced Records, Underground Resistance, Future Retro....", key="tab2_label")
            artist_search = st.text_input("🔍 Filter by artist", placeholder="e.g., Manix, DJ Rashad, Kid Lib....", key="tab2_artist")
            
            browse_df = filtered_df.copy()
            if label_search:
                browse_df = browse_df[browse_df['label'].str.lower().str.contains(label_search.lower(), na=False)]
            
            if artist_search:
                browse_df = browse_df[browse_df['parsed_artist'].str.lower().str.contains(artist_search.lower(), na=False)]
            
            if label_search or artist_search:
                st.info(f"Found {len(browse_df):,} releases matching filters")
            
            if len(browse_df) == 0:
                st.warning("No releases match your search. Try a different term.")
            else:
                page_size = st.selectbox("Results per page", [25, 50, 100, 200], index=2, key="tab2_pagesize")
                total_pages = max(1, (len(browse_df) - 1) // page_size + 1)
                page = st.number_input("Page", 1, total_pages, 1, key="tab2_page")
                
                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size
                
                page_df = browse_df.iloc[start_idx:end_idx]
                
                display_cols = ['title', 'oldest_year', 'label', 'country', 'style', 'resource_url']
                available_cols = [c for c in display_cols if c in page_df.columns]
                
                st.dataframe(
                    page_df[available_cols],
                    column_config={
                        "title": "Title",
                        "oldest_year": "Year",
                        "label": "Label",
                        "country": "Country",
                        "style": "Style",
                        "resource_url": st.column_config.LinkColumn("Discogs Link")
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                st.caption(f"Showing {start_idx+1}-{min(end_idx, len(browse_df))} of {len(browse_df)} releases")
    
    # ========================================
    # TAB 3: MY LIBRARY (XML UPLOAD)
    # ========================================
    with tab3:
        st.header("📁 My Library")
        st.markdown("Upload your Rekordbox XML to find mix-compatible tracks and discover related Discogs releases.")
        
        st.info("💡 **Tip:** For best Discogs recommendations, ensure your tracks have complete metadata (Artist, Label, Year, Genre).")
        
        st.warning("⚠️ **Hugging Face Users:** File uploads may not work on the hosted version. Please use **'Paste XML content'** instead. Both options work when running locally.")
        
        # Two options: file upload or paste
        upload_method = st.radio("Choose input method:", ["Upload XML file", "Paste XML content"], horizontal=True)
        
        xml_content = None
        
        if upload_method == "Upload XML file":
            uploaded_file = st.file_uploader("Upload Rekordbox XML", type=['xml'], key="xml_upload")
            if uploaded_file is not None:
                xml_content = uploaded_file.read().decode('utf-8')
        else:
            st.markdown("Paste your Rekordbox XML content below:")
            pasted_xml = st.text_area("XML Content", height=200, placeholder="<?xml version='1.0'...>", key="xml_paste")
            if pasted_xml and (pasted_xml.strip().startswith("<?xml") or pasted_xml.strip().startswith("<DJ_PLAYLISTS")):
                xml_content = pasted_xml
            elif pasted_xml:
                st.warning("This doesn't look like valid XML. Make sure it starts with <?xml or <DJ_PLAYLISTS")
        
        if xml_content:
            with st.spinner("Parsing XML..."):
                library_df = parse_rekordbox_xml(xml_content)
            
            if library_df is None or len(library_df) == 0:
                st.error("No tracks found in the XML. Please check the file format.")
            else:
                st.success(f"✅ Loaded {len(library_df)} tracks from your library")
                
                st.session_state['library_df'] = library_df
                
                st.subheader("Your Library")
                
                display_df = library_df.copy()
                display_df['#'] = range(1, len(display_df) + 1)
                display_df['BPM'] = display_df['bpm'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else 'N/A')
                display_df['Key'] = display_df['key_original'].apply(lambda x: x if x else 'N/A')
                display_df['Artist'] = display_df['artist'].apply(lambda x: x if x else 'N/A')
                display_df['Title'] = display_df['title'].apply(lambda x: x if x else 'N/A')
                display_df['Label'] = display_df['label'].apply(lambda x: x if x else 'N/A')
                display_df['Year'] = display_df['year'].apply(lambda x: str(int(x)) if pd.notna(x) else 'N/A')
                display_df['Genre'] = display_df['genre'].apply(lambda x: x if x else 'N/A')
                
                st.dataframe(
                    display_df[['#', 'Artist', 'Title', 'BPM', 'Key', 'Label', 'Year', 'Genre']],
                    hide_index=True,
                    use_container_width=True,
                    height=400
                )
                
                st.divider()
                
                st.subheader("Get Recommendations")
                
                track_options = [f"{i+1}. {row['artist']} - {row['title']}" for i, row in library_df.iterrows()]
                
                selected_track = st.selectbox("Select a track", track_options, key="xml_track_select")
                
                if selected_track:
                    selected_idx = int(selected_track.split('.')[0]) - 1
                    seed_track = library_df.iloc[selected_idx]
                    
                    st.markdown("**Selected Track:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Artist", seed_track['artist'] or 'N/A')
                    with col2:
                        st.metric("Title", seed_track['title'] or 'N/A')
                    with col3:
                        st.metric("BPM", f"{seed_track['bpm']:.1f}" if pd.notna(seed_track['bpm']) else 'N/A')
                    with col4:
                        st.metric("Key", seed_track['key_original'] or 'N/A')
                    
                    if st.button("🎵 Find Recommendations", key="xml_get_recs"):
                        st.session_state['xml_selected_idx'] = selected_idx
                
                if 'xml_selected_idx' in st.session_state:
                    selected_idx = st.session_state['xml_selected_idx']
                    seed_track = library_df.iloc[selected_idx]
                    
                    st.divider()
                    
                    st.subheader("🎧 Mix-Compatible Tracks (from your library)")
                    
                    if seed_track['bpm'] is None or seed_track['key_camelot'] is None:
                        st.warning("BPM or Key data missing for this track. Cannot find mix-compatible tracks.")
                    else:
                        with st.spinner("Finding mix-compatible tracks..."):
                            compatible = find_mix_compatible_tracks(library_df, selected_idx)
                        
                        if compatible:
                            st.success(f"Found {len(compatible)} mix-compatible tracks")
                            
                            for i, track in enumerate(compatible[:15], 1):
                                with st.container():
                                    col1, col2 = st.columns([4, 1])
                                    with col1:
                                        st.markdown(f"**{i}. {track['artist']} - {track['title']}**")
                                        st.caption(f"{track['reason']}")
                                    with col2:
                                        st.metric("Score", f"{track['score']:.2f}")
                                    st.divider()
                        else:
                            st.info("No mix-compatible tracks found in your library for this track.")
                    
                    st.divider()
                    st.subheader("💿 Related Discogs Releases")
                    
                    has_artist = bool(seed_track.get('artist') and str(seed_track['artist']).strip())
                    has_label = bool(seed_track.get('label') and str(seed_track['label']).strip())
                    has_year = bool(seed_track.get('year') and pd.notna(seed_track['year']))
                    has_genre = bool(seed_track.get('genre') and str(seed_track['genre']).strip())
                    
                    available_fields = []
                    if has_artist:
                        available_fields.append("Artist")
                    if has_label:
                        available_fields.append("Label")
                    if has_year:
                        available_fields.append("Year")
                    if has_genre:
                        available_fields.append("Genre")
                    
                    if not available_fields:
                        st.warning("⚠️ More tags required for Discogs recommendations. Please add metadata to your track.")
                    else:
                        st.caption(f"Matching using: {', '.join(available_fields)}")
                        
                        with st.spinner("Finding related Discogs releases..."):
                            top_matches, discovery = find_discogs_matches(seed_track, filtered_df, n=10)
                        
                        if top_matches:
                            st.markdown("**⭐ Top Matches**")
                            for i, match in enumerate(top_matches[:5], 1):
                                row = match['row']
                                score = match['score']
                                reasons = match['reasons']
                                
                                with st.container():
                                    col1, col2, col3 = st.columns([3, 1, 1])
                                    
                                    with col1:
                                        year = int(row.get('oldest_year', row['year'])) if pd.notna(row.get('oldest_year', row['year'])) else 'Unknown'
                                        st.markdown(f"**{i}. {row['title']}** ({year})")
                                        label_display = str(row['label'])[:50] + '...' if pd.notna(row['label']) and len(str(row['label'])) > 50 else row['label']
                                        st.caption(f"Label: {label_display}")
                                        if reasons:
                                            st.caption(f"💡 {', '.join(reasons[:5])}")
                                    
                                    with col2:
                                        st.metric("Score", f"{score:.2f}")
                                    
                                    with col3:
                                        if pd.notna(row.get('resource_url')):
                                            st.link_button("Discogs", row['resource_url'])
                                    
                                    st.divider()
                        else:
                            st.info("No matching Discogs releases found. Try adding more metadata to your track.")
                        
                        if discovery and has_label:
                            st.markdown("**🔮 Discovery Picks** (same label, different vibe)")
                            for i, match in enumerate(discovery, 1):
                                row = match['row']
                                score = match['score']
                                reasons = match['reasons']
                                
                                with st.container():
                                    col1, col2, col3 = st.columns([3, 1, 1])
                                    
                                    with col1:
                                        year = int(row.get('oldest_year', row['year'])) if pd.notna(row.get('oldest_year', row['year'])) else 'Unknown'
                                        st.markdown(f"**{i}. {row['title']}** ({year})")
                                        label_display = str(row['label'])[:50] + '...' if pd.notna(row['label']) and len(str(row['label'])) > 50 else row['label']
                                        st.caption(f"Label: {label_display}")
                                        if reasons:
                                            st.caption(f"💡 {', '.join(reasons[:5])}")
                                    
                                    with col2:
                                        st.metric("Score", f"{score:.2f}")
                                    
                                    with col3:
                                        if pd.notna(row.get('resource_url')):
                                            st.link_button("Discogs", row['resource_url'])
                                    
                                    st.divider()
        else:
            st.markdown("### How to export your Rekordbox library:")
            st.markdown("""
            1. Open Rekordbox
            2. Go to **File** → **Export Collection in xml format**
            3. Save the file and upload it here (or paste the contents)
            """)


if __name__ == "__main__":
    main()