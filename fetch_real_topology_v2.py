"""
fetch_real_topology_v2.py — Fetch REAL 2×2 grid from OSM using known roads.

Uses manually selected intersections from the Quartier Ganhi road network
where Avenue Augustin Nikoué and Avenue du Capitaine J. Adjovi form a grid
with perpendicular cross-streets.
"""

import osmnx as ox
import folium
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

ox.settings.overpass_rate_limit = False
ox.settings.timeout = 120

print("=" * 70)
print("FETCHING REAL OSM DATA — Quartier Ganhi, Cotonou (v2)")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# 1. FETCH NETWORK
# ═══════════════════════════════════════════════════════════════════════════
CENTER_LAT, CENTER_LON = 6.3605, 2.4270
print("\n[1/6] Downloading road network from OpenStreetMap...")
G = ox.graph_from_point((CENTER_LAT, CENTER_LON), dist=700, network_type='drive')
print(f"  → {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ═══════════════════════════════════════════════════════════════════════════
# 2. SELECT 4 REAL INTERSECTIONS FORMING A GRID
# ═══════════════════════════════════════════════════════════════════════════
# From the road exploration, we identify a clear grid pattern:
#
# Avenue du Capitaine J. Adjovi runs NW–SE (roughly N–S axis)
# Avenue Augustin Nikoué runs NW–SE (roughly parallel, slightly east)
# Cross streets: Rue d'Abomey, Rue de l'Afrique, Rue José Firmin Santos
#
# Best 2×2 grid using cross-streets as EW connectors:
#
#    I1 (Adjovi × Afrique)  ——  Rue de l'Afrique  ——  I2 (Nikoué × Afrique)
#           |                                                  |
#    Av. Adjovi                                        Av. Nikoué
#           |                                                  |
#    I3 (Adjovi × Abomey)   ——  Rue d'Abomey     ——  I4 (Nikoué × Abomey)
#

print("\n[2/6] Selecting real intersections...")

# Specific OSM node IDs from the exploration
# Using cross-streets Félix Éboué (NW) and José Firmin Santos (SE)
# for wider N-S spacing (~280m instead of ~70m)
SELECTED_NODES = {
    'I1': 2782446101,  # Av. Capitaine Adjovi × Rue Félix Éboué
    'I2': 2782446111,  # Av. Augustin Nikoué × Rue Félix Éboué
    'I3': 2782446087,  # Av. Capitaine Adjovi × Rue José Firmin Santos
    'I4': 2782446098,  # Av. Augustin Nikoué × Rue José Firmin Santos
}

# Verify all nodes exist in graph
for label, nid in SELECTED_NODES.items():
    assert nid in G.nodes, f"Node {nid} ({label}) not found in graph!"
    data = G.nodes[nid]
    print(f"  {label}: OSM node {nid} at ({data['y']:.6f}, {data['x']:.6f})")

# Road names at each junction
JUNCTION_ROADS = {
    'I1': "Av. Capitaine Adjovi × Rue Félix Éboué",
    'I2': "Av. Augustin Nikoué × Rue Félix Éboué",
    'I3': "Av. Capitaine Adjovi × Rue J. F. Santos",
    'I4': "Av. Augustin Nikoué × Rue J. F. Santos",
}

# ═══════════════════════════════════════════════════════════════════════════
# 3. COMPUTE REAL ROAD LENGTHS
# ═══════════════════════════════════════════════════════════════════════════

print("\n[3/6] Computing real road lengths along shortest paths...")

LINK_DEFS = {
    'L1': ('I1', 'I2', "Rue Félix Éboué", 'EW'),              # top EW
    'L2': ('I3', 'I4', "Rue José Firmin Santos", 'EW'),        # bottom EW
    'L3': ('I1', 'I3', "Av. Capitaine Adjovi", 'NS'),          # left NS
    'L4': ('I2', 'I4', "Av. Augustin Nikoué", 'NS'),           # right NS
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def get_path_road_names(G, path):
    names = set()
    for i in range(len(path) - 1):
        edge_data = G.get_edge_data(path[i], path[i+1])
        if edge_data:
            for key, data in edge_data.items():
                name = data.get('name')
                if name:
                    if isinstance(name, list):
                        names.update(name)
                    else:
                        names.add(name)
    return names

link_paths = {}
real_lengths = {}
real_road_names = {}

print(f"\n{'Link':<6} {'From→To':<10} {'Road (m)':<12} {'Haversine (m)':<16} {'Road Name'}")
print("-" * 80)

for lid, (from_j, to_j, expected_name, direction) in LINK_DEFS.items():
    src_id = SELECTED_NODES[from_j]
    dst_id = SELECTED_NODES[to_j]
    
    path = nx.shortest_path(G, src_id, dst_id, weight='length')
    road_len = nx.shortest_path_length(G, src_id, dst_id, weight='length')
    
    src = G.nodes[src_id]
    dst = G.nodes[dst_id]
    hav = haversine(src['y'], src['x'], dst['y'], dst['x'])
    
    path_names = get_path_road_names(G, path)
    actual_name = expected_name
    if path_names:
        actual_name = ' / '.join(sorted(str(n) for n in path_names if isinstance(n, str)))
    
    link_paths[lid] = {
        'path': path,
        'length': road_len,
        'haversine': hav,
        'from': from_j,
        'to': to_j,
        'direction': direction,
        'road_name': actual_name,
    }
    real_lengths[lid] = road_len
    real_road_names[lid] = actual_name
    
    n_cells = max(5, round(road_len / 30))
    dx = road_len / n_cells
    print(f"  {lid:<4} {from_j}→{to_j}  {road_len:>8.1f}    {hav:>10.1f}       {actual_name}")

total_len = sum(real_lengths.values())
print(f"\n  TOTAL: {total_len:.0f}m")

# ═══════════════════════════════════════════════════════════════════════════
# 4. GENERATE FOLIUM MAP
# ═══════════════════════════════════════════════════════════════════════════

print("\n[4/6] Generating interactive map...")

center_lat = np.mean([G.nodes[SELECTED_NODES[j]]['y'] for j in SELECTED_NODES])
center_lon = np.mean([G.nodes[SELECTED_NODES[j]]['x'] for j in SELECTED_NODES])

m = folium.Map(location=[center_lat, center_lon], zoom_start=17,
               tiles='CartoDB positron')

# Also add satellite tiles
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri', name='Satellite', overlay=False
).add_to(m)
folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
folium.LayerControl().add_to(m)

link_colors = {'L1': '#e74c3c', 'L2': '#3498db', 'L3': '#2ecc71', 'L4': '#f39c12'}
link_labels = {
    'L1': "Rue Félix Éboué (Top E-W)",
    'L2': "Rue José Firmin Santos (Bottom E-W)",
    'L3': "Av. Capitaine Adjovi (Left N-S)",
    'L4': "Av. Augustin Nikoué (Right N-S)",
}

for lid, info in link_paths.items():
    path_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in info['path']]
    folium.PolyLine(
        path_coords, weight=7, color=link_colors[lid], opacity=0.85,
        popup=f"<b>{lid}: {link_labels[lid]}</b><br>Length: {info['length']:.0f}m",
        tooltip=f"{lid}: {info['length']:.0f}m",
    ).add_to(m)

for jid, nid in SELECTED_NODES.items():
    data = G.nodes[nid]
    folium.CircleMarker(
        location=[data['y'], data['x']], radius=12,
        color='#2c3e50', fill=True, fill_color='#e67e22', fill_opacity=0.9,
        popup=f"<b>{jid}</b><br>{JUNCTION_ROADS[jid]}<br>OSM: {nid}",
        tooltip=f"{jid} — {JUNCTION_ROADS[jid]}",
    ).add_to(m)
    folium.Marker(
        location=[data['y'], data['x']],
        icon=folium.DivIcon(html=f'<div style="font-size:16px;font-weight:bold;color:#2c3e50;text-shadow:2px 2px white;">{jid}</div>'),
    ).add_to(m)

title_html = '''
<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
     z-index:9999;background:white;padding:10px 20px;border-radius:8px;
     box-shadow:0 2px 6px rgba(0,0,0,0.3);font-size:15px;font-weight:bold;">
     Réseau 2×2 — Quartier Ganhi, Cotonou (Topologie OSM réelle)
</div>'''
m.get_root().html.add_child(folium.Element(title_html))

legend_html = '<div style="position:fixed;bottom:20px;right:20px;z-index:9999;background:white;padding:12px;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.3);font-size:12px;"><b>Links:</b><br>'
for lid, color in link_colors.items():
    legend_html += f'<span style="color:{color};font-weight:bold;">━━</span> {lid}: {link_labels[lid]} ({real_lengths[lid]:.0f}m)<br>'
legend_html += '</div>'
m.get_root().html.add_child(folium.Element(legend_html))

out_dir = Path(__file__).parent.parent / 'images' / 'chapter3'
out_dir.mkdir(parents=True, exist_ok=True)

map_path = out_dir / 'fig_network_osm_map.html'
m.save(str(map_path))
print(f"  → Saved: {map_path}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. STATIC FIGURE FOR THE ARTICLE
# ═══════════════════════════════════════════════════════════════════════════

print("\n[5/6] Generating publication figure...")

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Background: full road network in light gray
for u, v, data in G.edges(data=True):
    ud, vd = G.nodes[u], G.nodes[v]
    ax.plot([ud['x'], vd['x']], [ud['y'], vd['y']], color='#dddddd', linewidth=0.6, zorder=1)

# Selected links
for lid, info in link_paths.items():
    path = info['path']
    lons = [G.nodes[n]['x'] for n in path]
    lats = [G.nodes[n]['y'] for n in path]
    ax.plot(lons, lats, color=link_colors[lid], linewidth=5, zorder=3, solid_capstyle='round',
            label=f"{lid}: {real_road_names[lid][:35]} ({info['length']:.0f}m)")

# Junctions
for jid, nid in SELECTED_NODES.items():
    data = G.nodes[nid]
    ax.plot(data['x'], data['y'], 'o', color='#e67e22',
            markersize=16, markeredgecolor='#2c3e50', markeredgewidth=2.5, zorder=5)
    
    # Position labels intelligently
    if 'I1' == jid:
        ha, va, ox_off, oy_off = 'right', 'bottom', -0.0003, 0.0002
    elif 'I2' == jid:
        ha, va, ox_off, oy_off = 'left', 'bottom', 0.0003, 0.0002
    elif 'I3' == jid:
        ha, va, ox_off, oy_off = 'right', 'top', -0.0003, -0.0002
    else:
        ha, va, ox_off, oy_off = 'left', 'top', 0.0003, -0.0002
    
    ax.annotate(f"{jid}\n{JUNCTION_ROADS[jid]}", (data['x'] + ox_off, data['y'] + oy_off),
                fontsize=8, fontweight='bold', color='#2c3e50', ha=ha, va=va,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2c3e50', alpha=0.9),
                zorder=6)

# Length annotations on link midpoints
for lid, info in link_paths.items():
    path = info['path']
    mid_idx = len(path) // 2
    mid_x = G.nodes[path[mid_idx]]['x']
    mid_y = G.nodes[path[mid_idx]]['y']
    
    # Offset perpendicular to link direction
    if info['direction'] == 'EW':
        oy = 0.0002
        ox = 0
    else:
        ox = 0.0003
        oy = 0
    
    ax.annotate(f"{info['length']:.0f} m", (mid_x + ox, mid_y + oy),
                fontsize=10, fontweight='bold', color=link_colors[lid],
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=link_colors[lid], alpha=0.9),
                zorder=7)

# Flow arrows
for lid, info in link_paths.items():
    path = info['path']
    if len(path) < 3:
        continue
    n1, n2 = len(path) // 3, 2 * len(path) // 3
    x1, y1 = G.nodes[path[n1]]['x'], G.nodes[path[n1]]['y']
    x2, y2 = G.nodes[path[n2]]['x'], G.nodes[path[n2]]['y']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=link_colors[lid], lw=2.5),
                zorder=4)

ax.set_xlabel('Longitude', fontsize=11)
ax.set_ylabel('Latitude', fontsize=11)
ax.set_title('2×2 Grid Network — Quartier Ganhi, Cotonou\n(Real OpenStreetMap Topology)', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=8.5, framealpha=0.9)

# Scale bar (100m)
scale_deg = 100 / (111320 * np.cos(np.radians(6.36)))
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
bar_x = xmin + 0.05 * (xmax - xmin)
bar_y = ymin + 0.06 * (ymax - ymin)
ax.plot([bar_x, bar_x + scale_deg], [bar_y, bar_y], 'k-', linewidth=3)
ax.text(bar_x + scale_deg/2, bar_y + 0.00008, '100 m', ha='center', fontsize=9, fontweight='bold')

# North arrow
arrow_x = xmax - 0.06 * (xmax - xmin)
arrow_y = ymax - 0.06 * (ymax - ymin)
ax.annotate('N', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y - 0.0004),
            fontsize=13, fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', color='black', lw=2.5))

ax.set_aspect('equal')
plt.tight_layout()

for ext in ['png', 'pdf']:
    fp = out_dir / f'fig_network_real_osm.{ext}'
    fig.savefig(str(fp), dpi=600, bbox_inches='tight')
    print(f"  → Saved: {fp}")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# 6. EXPORT TOPOLOGY DATA
# ═══════════════════════════════════════════════════════════════════════════

print("\n[6/6] Exporting topology data...")

topo = {
    'area': 'Quartier Ganhi, Cotonou, Benin',
    'source': 'OpenStreetMap via osmnx',
    'junctions': {},
    'links': {},
}

for jid, nid in SELECTED_NODES.items():
    data = G.nodes[nid]
    topo['junctions'][jid] = {
        'osm_id': int(nid),
        'lat': float(data['y']),
        'lon': float(data['x']),
        'roads': JUNCTION_ROADS[jid],
    }

for lid in ['L1', 'L2', 'L3', 'L4']:
    info = link_paths[lid]
    n_cells = max(5, round(info['length'] / 30))
    topo['links'][lid] = {
        'from': info['from'],
        'to': info['to'],
        'length_m': round(info['length'], 1),
        'n_cells': n_cells,
        'dx_m': round(info['length'] / n_cells, 1),
        'direction': info['direction'],
        'road_name': real_road_names[lid],
    }

json_path = out_dir / 'network_topology.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(topo, f, indent=2, ensure_ascii=False)
print(f"  → Saved: {json_path}")

# Print summary for network_params.py update
print("\n" + "=" * 70)
print("VALUES FOR network_params.py:")
print("=" * 70)
for lid in ['L1', 'L2', 'L3', 'L4']:
    info = link_paths[lid]
    n_cells = max(5, round(info['length'] / 30))
    dx = info['length'] / n_cells
    print(f"  '{lid}': Link('{lid}', '{info['from']}', '{info['to']}', {info['length']:.1f}, {n_cells}, '{real_road_names[lid][:30]}', '{info['direction']}'),")

total_cells = sum(max(5, round(link_paths[lid]['length'] / 30)) for lid in ['L1', 'L2', 'L3', 'L4'])
print(f"\n  TOTAL_CELLS = {total_cells}")
print(f"  TOTAL_LENGTH = {total_len:.0f}m")
print("\n✓ Done!")
