"""
INTELLIGENT CRATE DIGGING - STREAMLIT APP
A hybrid AI recommendation system for UK jungle/hardcore DJs

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import networkx as nx
import pickle
import re
import random
import os
from collections import defaultdict

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

# === EDIT THESE PATHS ===
BASE_PATH = "/Users/benhasy/Documents/UNI/Foundations of AI/api/PROCESSING_CSVS"
COMBINED_PATH = os.path.join(BASE_PATH, "_combined")

# Method weights
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
        ' x ', ' X ', ' & ', ' and ', ', ', ','
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
        # Parse artists if needed
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
# RECOMMENDATION ENGINE
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
        
        best_score = 0
        best_reason = []
        
        for s_artist in seed_artists:
            if s_artist not in self.graph:
                continue
            for c_artist in cand_artists:
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
        seed_artists = set(a.lower() for a in (seed_row['artists_list'] if isinstance(seed_row['artists_list'], list) else []))
        
        candidates = []
        
        for idx, candidate_row in self.df.iterrows():
            if idx == seed_idx:
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
            
            if final_score > 0:
                candidates.append({
                    'idx': idx,
                    'row': candidate_row,
                    'score': final_score,
                    'breakdown': scores,
                    'reasons': all_reasons
                })
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Top 3 stay, rest shuffled
        top_3 = candidates[:3]
        rest = candidates[3:n]
        random.shuffle(rest)
        
        return top_3 + rest


# ============================================================
# STREAMLIT UI
# ============================================================

def main():
    st.title("🎵 Intelligent Crate Digging")
    st.markdown("*A hybrid AI recommendation system for UK jungle/hardcore DJs*")
    
    # Sidebar
    st.sidebar.header("🔧 Settings")
    
    # Format selection
    formats = get_available_formats()
    if not formats:
        st.error(f"No data found in {COMBINED_PATH}. Please check the BASE_PATH.")
        return
    
    selected_format = st.sidebar.selectbox(
        "Select Format",
        formats,
        index=formats.index('vinyl') if 'vinyl' in formats else 0
    )
    
    # Load data
    with st.spinner(f"Loading {selected_format} data..."):
        df = load_combined_csv(selected_format)
    
    if df is None:
        st.error(f"Could not load data for {selected_format}")
        return
    
    st.sidebar.success(f"Loaded {len(df):,} releases")
    
    # Filters
    st.sidebar.header("🎚️ Filters")
    
    # Year range
    min_year = int(df['oldest_year'].min()) if df['oldest_year'].min() > 0 else 1990
    max_year = int(df['oldest_year'].max())
    year_range = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))
    
    # Country filter
    countries = ['All'] + sorted(df['country'].dropna().unique().tolist())
    selected_country = st.sidebar.selectbox("Country", countries)
    
    # Style filter
    all_styles = set()
    for styles in df['style'].dropna():
        for s in str(styles).split(','):
            all_styles.add(s.strip())
    style_options = ['All'] + sorted(all_styles)
    selected_style = st.sidebar.selectbox("Style", style_options)
    
    # Apply filters
    filtered_df = df.copy()
    filtered_df = filtered_df[(filtered_df['oldest_year'] >= year_range[0]) & 
                               (filtered_df['oldest_year'] <= year_range[1])]
    
    if selected_country != 'All':
        filtered_df = filtered_df[filtered_df['country'] == selected_country]
    
    if selected_style != 'All':
        filtered_df = filtered_df[filtered_df['style'].str.contains(selected_style, case=False, na=False)]
    
    st.sidebar.info(f"Showing {len(filtered_df):,} releases after filters")
    
    # Exclude same artist option
    exclude_same_artist = st.sidebar.checkbox("Exclude same artist from recommendations", value=False)
    
    # Main content
    tab1, tab2 = st.tabs(["🔍 Search & Recommend", "📋 Browse All"])
    
    # ========================================
    # TAB 1: SEARCH & RECOMMEND
    # ========================================
    with tab1:
        st.header("Search for a Release")
        
        search_query = st.text_input("Search by artist or title", placeholder="e.g., DJ Hype, Renegade, Moving Shadow...")
        
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
                
                # Display results
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
        
        # Show recommendations if a release is selected
        if 'selected_release_idx' in st.session_state:
            st.divider()
            st.header(f"Recommendations for: {st.session_state['selected_release_title']}")
            
            # Get the actual index in filtered_df
            selected_idx = st.session_state['selected_release_idx']
            
            # Create engine with filtered data
            engine = RecommendationEngine(filtered_df, graph=None)
            
            # Find the position in filtered_df
            if selected_idx in filtered_df.index:
                position = filtered_df.index.get_loc(selected_idx)
                
                with st.spinner("Finding recommendations..."):
                    recommendations = engine.get_recommendations(position, n=50, exclude_same_artist=exclude_same_artist)
                
                if recommendations:
                    st.success(f"Found {len(recommendations)} recommendations")
                    
                    # Number of results to show
                    n_display = st.slider("Show top N results", 10, 50, 20)
                    
                    for i, rec in enumerate(recommendations[:n_display], 1):
                        row = rec['row']
                        score = rec['score']
                        reasons = rec['reasons']
                        
                        with st.container():
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                year = int(row.get('oldest_year', row['year'])) if pd.notna(row.get('oldest_year', row['year'])) else 'Unknown'
                                
                                # Highlight top 3
                                if i <= 3:
                                    st.markdown(f"### {i}. {row['title']} ({year}) ⭐")
                                else:
                                    st.markdown(f"**{i}. {row['title']}** ({year})")
                                
                                label_display = str(row['label'])[:50] + '...' if pd.notna(row['label']) and len(str(row['label'])) > 50 else row['label']
                                st.caption(f"Label: {label_display}")
                                
                                # Reasons
                                if reasons:
                                    st.caption(f"💡 {', '.join(reasons[:3])}")
                            
                            with col2:
                                st.metric("Score", f"{score:.2f}")
                            
                            with col3:
                                if pd.notna(row.get('resource_url')):
                                    st.link_button("View on Discogs", row['resource_url'])
                            
                            st.divider()
                else:
                    st.warning("No recommendations found for this release.")
            else:
                st.warning("Please search and select a release first.")
    
    # ========================================
    # TAB 2: BROWSE ALL
    # ========================================
    with tab2:
        st.header("Browse Database")
        
        # Pagination
        page_size = st.selectbox("Results per page", [25, 50, 100], index=1)
        total_pages = (len(filtered_df) - 1) // page_size + 1
        page = st.number_input("Page", 1, total_pages, 1)
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        page_df = filtered_df.iloc[start_idx:end_idx]
        
        # Display table
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
        
        st.caption(f"Showing {start_idx+1}-{min(end_idx, len(filtered_df))} of {len(filtered_df)} releases")


if __name__ == "__main__":
    main()
# End of streamlit_app_old_version.py