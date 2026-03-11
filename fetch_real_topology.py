#!/usr/bin/env python3
"""
fetch_real_topology.py — Fetch REAL road topology from OpenStreetMap
for a 2×2 grid of intersections near Quartier Ganhi / Dantokpa, Cotonou.

Produces:
  1. A folium interactive map (HTML) with the selected intersections & links
  2. A static matplotlib figure for the article (PNG + PDF)
  3. Prints the real segment lengths for updating network_params.py
"""

import osmnx as ox
import folium
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# osmnx settings for reliability
ox.settings.overpass_rate_limit = False
ox.settings.timeout = 120

# ═══════════════════════════════════════════════════════════════════════════
# 1. TARGET AREA: Quartier Ganhi / Dantokpa, Cotonou, Benin
# ═══════════════════════════════════════════════════════════════════════════
# Center of the Dantokpa market area
CENTER_LAT = 6.3625
CENTER_LON = 2.4279
DIST = 600  # meters radius to fetch

print("=" * 70)
print("FETCHING REAL OSM DATA — Quartier Ganhi, Cotonou")
print("=" * 70)

# Fetch the drivable road network
print("\n[1/5] Downloading road network from OpenStreetMap...")
G = ox.graph_from_point((CENTER_LAT, CENTER_LON), dist=DIST, network_type='drive')
print(f"  → {G.number_of_nodes()} nodes, {G.number_of_edges()} edges fetched")

# ═══════════════════════════════════════════════════════════════════════════
# 2. IDENTIFY KEY INTERSECTIONS
# ═══════════════════════════════════════════════════════════════════════════
# We want a 2×2 grid of real intersections connected by real roads.
# Strategy: find intersections (degree >= 3) and pick a suitable 2×2 group.

print("\n[2/5] Identifying signalized intersections...")

# Get node attributes
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

# Find intersections (degree >= 3)
intersections = []
for node, data in G.nodes(data=True):
    deg = G.degree(node)
    if deg >= 3:
        intersections.append({
            'id': node,
            'lat': data['y'],
            'lon': data['x'],
            'degree': deg,
        })

print(f"  → {len(intersections)} intersections (degree ≥ 3) found")

# Keep only the top intersections by degree (limit combinatorial explosion)
intersections.sort(key=lambda n: -n['degree'])
intersections = intersections[:40]  # top 40 by connectivity
print(f"  → Using top {len(intersections)} intersections by degree for grid search")

# Sort by latitude (N→S) then longitude (W→E) to form a grid
intersections.sort(key=lambda n: (-n['lat'], n['lon']))

# We need to find 4 intersections that form roughly a 2×2 grid
# Strategy: find pairs of nodes that are ~200-500m apart and form a grid

from itertools import combinations
import networkx as nx

def haversine(lat1, lon1, lat2, lon2):
    """Distance in meters between two points."""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# Build distance matrix for intersections
n_int = len(intersections)
dist_matrix = np.zeros((n_int, n_int))
for i in range(n_int):
    for j in range(n_int):
        dist_matrix[i, j] = haversine(
            intersections[i]['lat'], intersections[i]['lon'],
            intersections[j]['lat'], intersections[j]['lon']
        )

# Find best 2×2 grid: 4 nodes where each pair has a road path
# and the geometry roughly forms a rectangle
best_score = float('inf')
best_quad = None
best_info = None

MIN_SIDE = 150   # minimum edge ~150m
MAX_SIDE = 500   # maximum edge ~500m

for combo in combinations(range(n_int), 4):
    pts = [intersections[i] for i in combo]
    lats = [p['lat'] for p in pts]
    lons = [p['lon'] for p in pts]

    # Sort into grid: NW, NE, SW, SE
    # Sort by lat (desc) first, then lon (asc) within rows
    sorted_pts = sorted(pts, key=lambda p: (-p['lat'], p['lon']))
    nw, ne = sorted_pts[0], sorted_pts[1]
    sw, se = sorted_pts[2], sorted_pts[3]

    # Check that NW is actually west of NE, and SW is west of SE
    if nw['lon'] > ne['lon'] or sw['lon'] > se['lon']:
        continue

    # Compute the 4 side lengths and 2 diagonals
    d_top = haversine(nw['lat'], nw['lon'], ne['lat'], ne['lon'])
    d_bot = haversine(sw['lat'], sw['lon'], se['lat'], se['lon'])
    d_left = haversine(nw['lat'], nw['lon'], sw['lat'], sw['lon'])
    d_right = haversine(ne['lat'], ne['lon'], se['lat'], se['lon'])

    sides = [d_top, d_bot, d_left, d_right]

    # All sides within range
    if not all(MIN_SIDE <= s <= MAX_SIDE for s in sides):
        continue

    # Check connectivity via road network (all 4 edges of the grid exist)
    node_ids = [p['id'] for p in [nw, ne, sw, se]]
    edges_to_check = [
        (nw['id'], ne['id']),  # top
        (sw['id'], se['id']),  # bottom
        (nw['id'], sw['id']),  # left
        (ne['id'], se['id']),  # right
    ]

    all_connected = True
    path_lengths = []
    for u, v in edges_to_check:
        try:
            path_len = nx.shortest_path_length(G, u, v, weight='length')
            path_lengths.append(path_len)
            # Path shouldn't be much longer than haversine (max 2× for urban roads)
            hav = haversine(
                G.nodes[u]['y'], G.nodes[u]['x'],
                G.nodes[v]['y'], G.nodes[v]['x']
            )
            if path_len > 3.0 * hav:
                all_connected = False
                break
        except nx.NetworkXNoPath:
            all_connected = False
            break

    if not all_connected:
        continue

    # Score: prefer near-rectangular shape (sides similar, close to haversine)
    side_var = np.std(sides) / np.mean(sides)  # lower = more regular
    path_ratio = np.mean([pl / s for pl, s in zip(path_lengths, sides)])  # closer to 1 = more direct
    score = side_var + 0.5 * (path_ratio - 1.0)

    if score < best_score:
        best_score = score
        best_quad = {
            'NW': nw, 'NE': ne, 'SW': sw, 'SE': se,
            'path_lengths': path_lengths,
            'haversine_sides': sides,
        }
        best_info = {
            'top': (nw, ne, path_lengths[0]),
            'bottom': (sw, se, path_lengths[1]),
            'left': (nw, sw, path_lengths[2]),
            'right': (ne, se, path_lengths[3]),
        }

if best_quad is None:
    print("  ⚠ No suitable 2×2 grid found with current constraints.")
    print("  → Relaxing constraints and trying again...")
    MIN_SIDE = 100
    MAX_SIDE = 700
    for combo in combinations(range(min(n_int, 30)), 4):
        pts = [intersections[i] for i in combo]
        sorted_pts = sorted(pts, key=lambda p: (-p['lat'], p['lon']))
        nw, ne = sorted_pts[0], sorted_pts[1]
        sw, se = sorted_pts[2], sorted_pts[3]
        if nw['lon'] > ne['lon'] or sw['lon'] > se['lon']:
            continue
        d_top = haversine(nw['lat'], nw['lon'], ne['lat'], ne['lon'])
        d_bot = haversine(sw['lat'], sw['lon'], se['lat'], se['lon'])
        d_left = haversine(nw['lat'], nw['lon'], sw['lat'], sw['lon'])
        d_right = haversine(ne['lat'], ne['lon'], se['lat'], se['lon'])
        sides = [d_top, d_bot, d_left, d_right]
        if not all(MIN_SIDE <= s <= MAX_SIDE for s in sides):
            continue
        node_ids = [p['id'] for p in [nw, ne, sw, se]]
        edges_to_check = [
            (nw['id'], ne['id']),
            (sw['id'], se['id']),
            (nw['id'], sw['id']),
            (ne['id'], se['id']),
        ]
        all_connected = True
        path_lengths = []
        for u, v in edges_to_check:
            try:
                path_len = nx.shortest_path_length(G, u, v, weight='length')
                path_lengths.append(path_len)
                hav = haversine(
                    G.nodes[u]['y'], G.nodes[u]['x'],
                    G.nodes[v]['y'], G.nodes[v]['x']
                )
                if path_len > 3.0 * hav:
                    all_connected = False
                    break
            except nx.NetworkXNoPath:
                all_connected = False
                break
        if not all_connected:
            continue
        side_var = np.std(sides) / np.mean(sides)
        path_ratio = np.mean([pl / s for pl, s in zip(path_lengths, sides)])
        score = side_var + 0.5 * (path_ratio - 1.0)
        if score < best_score:
            best_score = score
            best_quad = {
                'NW': nw, 'NE': ne, 'SW': sw, 'SE': se,
                'path_lengths': path_lengths,
                'haversine_sides': sides,
            }
            best_info = {
                'top': (nw, ne, path_lengths[0]),
                'bottom': (sw, se, path_lengths[1]),
                'left': (nw, sw, path_lengths[2]),
                'right': (ne, se, path_lengths[3]),
            }

assert best_quad is not None, "Could not find a valid 2×2 grid. Try adjusting CENTER or DIST."

# ═══════════════════════════════════════════════════════════════════════════
# 3. EXTRACT REAL ROAD PATHS AND LENGTHS
# ═══════════════════════════════════════════════════════════════════════════

print("\n[3/5] Extracting real road paths and lengths...")

JUNCTION_MAP = {
    'I1 (NW)': best_quad['NW'],
    'I2 (NE)': best_quad['NE'],
    'I3 (SW)': best_quad['SW'],
    'I4 (SE)': best_quad['SE'],
}

# Get actual shortest paths along roads
link_paths = {}
link_defs = {
    'L1': ('NW', 'NE', 'Top (E-W)'),
    'L2': ('SW', 'SE', 'Bottom (E-W)'),
    'L3': ('NW', 'SW', 'Left (N-S)'),
    'L4': ('NE', 'SE', 'Right (N-S)'),
}

# Get road names for each path
def get_road_names(G, path):
    """Extract road names along a path."""
    names = set()
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_data = G.get_edge_data(u, v)
        if edge_data:
            for key, data in edge_data.items():
                name = data.get('name', None)
                if name:
                    if isinstance(name, list):
                        names.update(name)
                    else:
                        names.add(name)
    return names

print(f"\n{'Link':<6} {'From→To':<12} {'Road Length (m)':<18} {'Haversine (m)':<16} {'Road Names'}")
print("-" * 90)

real_lengths = {}
real_road_names = {}

for lid, (from_corner, to_corner, desc) in link_defs.items():
    src = best_quad[from_corner]
    dst = best_quad[to_corner]

    path = nx.shortest_path(G, src['id'], dst['id'], weight='length')
    road_len = nx.shortest_path_length(G, src['id'], dst['id'], weight='length')
    hav_dist = haversine(src['lat'], src['lon'], dst['lat'], dst['lon'])
    road_names = get_road_names(G, path)

    link_paths[lid] = {
        'path': path,
        'length': road_len,
        'haversine': hav_dist,
        'road_names': road_names,
        'desc': desc,
        'from': from_corner,
        'to': to_corner,
    }
    real_lengths[lid] = road_len
    real_road_names[lid] = ', '.join(road_names) if road_names else f"Unnamed ({desc})"
    
    print(f"  {lid:<4}  {from_corner}→{to_corner}   {road_len:>10.1f}m     {hav_dist:>10.1f}m     {real_road_names[lid]}")

total_len = sum(real_lengths.values())
print(f"\n  TOTAL network length: {total_len:.0f}m")

# ═══════════════════════════════════════════════════════════════════════════
# 4. GENERATE FOLIUM MAP
# ═══════════════════════════════════════════════════════════════════════════

print("\n[4/5] Generating interactive map...")

# Center of the 4 junctions
center_lat = np.mean([best_quad[k]['lat'] for k in ['NW', 'NE', 'SW', 'SE']])
center_lon = np.mean([best_quad[k]['lon'] for k in ['NW', 'NE', 'SW', 'SE']])

m = folium.Map(location=[center_lat, center_lon], zoom_start=16,
               tiles='CartoDB positron')

# Draw road paths for each link
link_colors = {'L1': '#e74c3c', 'L2': '#3498db', 'L3': '#2ecc71', 'L4': '#f39c12'}

for lid, info in link_paths.items():
    path_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in info['path']]
    folium.PolyLine(
        path_coords,
        weight=6,
        color=link_colors[lid],
        opacity=0.85,
        popup=f"<b>{lid}</b>: {real_road_names[lid]}<br>{info['length']:.0f}m",
        tooltip=f"{lid}: {info['length']:.0f}m",
    ).add_to(m)

# Mark junctions
junction_labels = {'NW': 'I1', 'NE': 'I2', 'SW': 'I3', 'SE': 'I4'}
for corner, node in best_quad.items():
    if corner in junction_labels:
        folium.CircleMarker(
            location=[node['lat'], node['lon']],
            radius=10,
            color='#2c3e50',
            fill=True,
            fill_color='#e67e22',
            fill_opacity=0.9,
            popup=f"<b>{junction_labels[corner]}</b> ({corner})<br>OSM Node: {node['id']}<br>Degree: {node['degree']}",
            tooltip=f"{junction_labels[corner]} — OSM #{node['id']}",
        ).add_to(m)
        # Add label
        folium.Marker(
            location=[node['lat'], node['lon']],
            icon=folium.DivIcon(html=f'<div style="font-size:14px;font-weight:bold;color:#2c3e50;text-shadow:1px 1px white;">{junction_labels[corner]}</div>'),
        ).add_to(m)

# Add title
title_html = '''
<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
     z-index:9999;background:white;padding:8px 16px;border-radius:8px;
     box-shadow:0 2px 6px rgba(0,0,0,0.3);font-size:14px;font-weight:bold;">
     2×2 Grid — Quartier Ganhi, Cotonou (Real OSM Topology)
</div>'''
m.get_root().html.add_child(folium.Element(title_html))

# Add legend
legend_html = '''
<div style="position:fixed;bottom:20px;right:20px;z-index:9999;background:white;
     padding:10px;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.3);font-size:12px;">
     <b>Links:</b><br>'''
for lid, color in link_colors.items():
    name = real_road_names[lid][:30]
    length = real_lengths[lid]
    legend_html += f'<span style="color:{color};font-weight:bold;">━━</span> {lid}: {name} ({length:.0f}m)<br>'
legend_html += '</div>'
m.get_root().html.add_child(folium.Element(legend_html))

# Save
out_dir = Path(__file__).parent.parent / 'images' / 'chapter3'
out_dir.mkdir(parents=True, exist_ok=True)

map_path = out_dir / 'fig_network_osm_map.html'
m.save(str(map_path))
print(f"  → Saved interactive map: {map_path}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. GENERATE STATIC FIGURE FOR ARTICLE
# ═══════════════════════════════════════════════════════════════════════════

print("\n[5/5] Generating publication figure...")

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot the full road network as background
for u, v, data in G.edges(data=True):
    u_data, v_data = G.nodes[u], G.nodes[v]
    ax.plot([u_data['x'], v_data['x']], [u_data['y'], v_data['y']],
            color='#cccccc', linewidth=0.8, zorder=1)

# Plot the selected links with color
for lid, info in link_paths.items():
    path = info['path']
    lons = [G.nodes[n]['x'] for n in path]
    lats = [G.nodes[n]['y'] for n in path]
    ax.plot(lons, lats, color=link_colors[lid], linewidth=4, zorder=3,
            label=f"{lid}: {real_road_names[lid][:25]} ({info['length']:.0f}m)")

# Plot junctions
for corner, node in best_quad.items():
    if corner in junction_labels:
        ax.plot(node['lon'], node['lat'], 'o', color='#e67e22',
                markersize=14, markeredgecolor='#2c3e50', markeredgewidth=2, zorder=5)
        # Label slightly offset
        offset_y = 0.0003 if 'N' in corner else -0.0004
        offset_x = -0.0003 if 'W' in corner else 0.0003
        ax.annotate(junction_labels[corner],
                    (node['lon'] + offset_x, node['lat'] + offset_y),
                    fontsize=13, fontweight='bold', color='#2c3e50',
                    ha='center', va='center', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))

# Add length annotations on links
for lid, info in link_paths.items():
    path = info['path']
    mid_idx = len(path) // 2
    mid_lon = G.nodes[path[mid_idx]]['x']
    mid_lat = G.nodes[path[mid_idx]]['y']
    ax.annotate(f"{info['length']:.0f}m",
                (mid_lon, mid_lat), fontsize=9, color=link_colors[lid],
                fontweight='bold', ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor=link_colors[lid], alpha=0.85),
                zorder=7)

# Flow direction arrows
for lid, info in link_paths.items():
    path = info['path']
    n1, n2 = len(path) // 3, 2 * len(path) // 3
    dx = G.nodes[path[n2]]['x'] - G.nodes[path[n1]]['x']
    dy = G.nodes[path[n2]]['y'] - G.nodes[path[n1]]['y']
    mid_x = G.nodes[path[n1]]['x'] + dx * 0.5
    mid_y = G.nodes[path[n1]]['y'] + dy * 0.5
    ax.annotate('', xy=(mid_x + dx * 0.15, mid_y + dy * 0.15),
                xytext=(mid_x - dx * 0.15, mid_y - dy * 0.15),
                arrowprops=dict(arrowstyle='->', color=link_colors[lid], lw=2),
                zorder=4)

ax.set_xlabel('Longitude', fontsize=11)
ax.set_ylabel('Latitude', fontsize=11)
ax.set_title('2×2 Grid Network — Quartier Ganhi, Cotonou\n(Real OpenStreetMap Topology)', fontsize=13, fontweight='bold')
ax.legend(loc='lower left', fontsize=9, framealpha=0.9)

# Add scale bar
from matplotlib.lines import Line2D
# approx 100m in degrees at lat 6.36
scale_deg = 100 / (111320 * np.cos(np.radians(6.36)))
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
bar_x = xmin + 0.05 * (xmax - xmin)
bar_y = ymin + 0.05 * (ymax - ymin)
ax.plot([bar_x, bar_x + scale_deg], [bar_y, bar_y], 'k-', linewidth=3)
ax.text(bar_x + scale_deg / 2, bar_y + 0.0001, '100m', ha='center', fontsize=9, fontweight='bold')

# Add north arrow
arrow_x = xmax - 0.05 * (xmax - xmin)
arrow_y = ymax - 0.05 * (ymax - ymin)
ax.annotate('N', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y - 0.0004),
            fontsize=12, fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

ax.set_aspect('equal')
plt.tight_layout()

# Save
for ext in ['png', 'pdf']:
    fig_path = out_dir / f'fig_network_real_osm.{ext}'
    fig.savefig(str(fig_path), dpi=600, bbox_inches='tight')
    print(f"  → Saved: {fig_path}")

plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# 6. OUTPUT SUMMARY FOR NETWORK PARAMS UPDATE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SUMMARY — Use these values to update network_params.py")
print("=" * 70)

print("\nJunction coordinates (lat, lon):")
for corner in ['NW', 'NE', 'SW', 'SE']:
    node = best_quad[corner]
    print(f"  {junction_labels[corner]} ({corner}): ({node['lat']:.6f}, {node['lon']:.6f})  [OSM node {node['id']}]")

print(f"\nLink real lengths and road names:")
for lid in ['L1', 'L2', 'L3', 'L4']:
    info = link_paths[lid]
    n_cells = max(5, round(info['length'] / 30))  # ~30m per cell
    dx = info['length'] / n_cells
    print(f"  {lid}: {info['length']:>7.1f}m  ({n_cells} cells × {dx:.1f}m)  — {real_road_names[lid]}")

print(f"\n  Total network length: {total_len:.0f}m")
print(f"  Total cells: {sum(max(5, round(link_paths[lid]['length'] / 30)) for lid in ['L1', 'L2', 'L3', 'L4'])}")

# Save the topology data as JSON for later use
topo_data = {
    'center': {'lat': CENTER_LAT, 'lon': CENTER_LON},
    'junctions': {},
    'links': {},
}
for corner in ['NW', 'NE', 'SW', 'SE']:
    node = best_quad[corner]
    topo_data['junctions'][junction_labels[corner]] = {
        'corner': corner,
        'lat': node['lat'],
        'lon': node['lon'],
        'osm_id': int(node['id']),
        'degree': node['degree'],
    }
for lid in ['L1', 'L2', 'L3', 'L4']:
    info = link_paths[lid]
    n_cells = max(5, round(info['length'] / 30))
    topo_data['links'][lid] = {
        'from': info['from'],
        'to': info['to'],
        'length_m': round(info['length'], 1),
        'haversine_m': round(info['haversine'], 1),
        'n_cells': n_cells,
        'road_names': list(info['road_names']),
    }

json_path = out_dir / 'network_topology.json'
with open(json_path, 'w') as f:
    json.dump(topo_data, f, indent=2)
print(f"\n  → Topology data saved: {json_path}")

print("\n✓ Done!")
