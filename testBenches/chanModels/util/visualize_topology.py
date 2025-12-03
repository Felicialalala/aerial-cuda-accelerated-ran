# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visualize network topology from YAML or H5 file"""

import yaml
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from matplotlib.patches import RegularPolygon
from collections import defaultdict


def parse_h5_topology(h5_file_path):
    """Parse topology data from H5 file"""
    with h5py.File(h5_file_path, 'r') as f:
        # Parse topology parameters
        if 'topology' in f:
            topo_group = f['topology']
            topology = {
                'ISD': float(topo_group['ISD'][()]),
                'nSite': int(topo_group['nSite'][()]),
                'nSector': int(topo_group['nSector'][()]),
                'nUT': int(topo_group['nUT'][()]),
                'bsHeight': float(topo_group['bsHeight'][()]),
                'minBsUeDist2d': float(topo_group['minBsUeDist2d'][()]),
                'maxBsUeDist2dIndoor': float(topo_group['maxBsUeDist2dIndoor'][()]),
                'indoorUtPercent': float(topo_group['indoorUtPercent'][()])
            }
            
            # Parse cell parameters (base stations)
            base_stations = []
            if 'cellParams' in topo_group:
                cell_dataset = topo_group['cellParams']
                for record in cell_dataset:
                    # Handle nested location structure
                    if hasattr(record['loc'], 'dtype') and record['loc'].dtype.names:
                        # Structured array
                        loc = record['loc']
                        location = {'x': float(loc['x']), 'y': float(loc['y']), 'z': float(loc['z'])}
                    else:
                        # Simple array [x, y, z]
                        loc = record['loc']
                        location = {'x': float(loc[0]), 'y': float(loc[1]), 'z': float(loc[2])}
                    
                    # Calculate sector orientation based on cell ID and site
                    cid = int(record['cid'])
                    site_id = int(record['siteId'])
                    sector_id = cid % 3  # Assuming 3 sectors per site
                    orientation = sector_id * 120  # 120 degrees apart
                    
                    base_stations.append({
                        'cid': cid,
                        'siteId': site_id,
                        'location': location,
                        'antPanelIdx': int(record['antPanelIdx']),
                        'antPanelOrientation': [0, orientation]  # [tilt, azimuth]
                    })
            
            # Parse UT parameters (user equipment)
            user_equipment = []
            if 'utParams' in topo_group:
                ut_group = topo_group['utParams']
                # Read arrays from the group
                uids = ut_group['uid'][:]
                locs_x = ut_group['loc_x'][:]
                locs_y = ut_group['loc_y'][:]
                locs_z = ut_group['loc_z'][:]
                outdoor_inds = ut_group['outdoor_ind'][:]
                ant_panel_idxs = ut_group['antPanelIdx'][:]
                
                for i in range(len(uids)):
                    user_equipment.append({
                        'uid': int(uids[i]),
                        'location': {
                            'x': float(locs_x[i]),
                            'y': float(locs_y[i]),
                            'z': float(locs_z[i])
                        },
                        'outdoor_ind': int(outdoor_inds[i]),
                        'antPanelIdx': int(ant_panel_idxs[i])
                    })
        else:
            raise ValueError("No topology data found in H5 file")
    
    return {
        'topology': topology,
        'base_stations': base_stations,
        'user_equipment': user_equipment
    }

def visualize_topology(input_file):
    """Visualize topology from either YAML or H5 file"""
    file_path = Path(input_file)
    
    # Determine file type and parse accordingly
    if file_path.suffix.lower() in ['.yaml', '.yml']:
        # Read YAML file
        with open(input_file, 'r') as f:
            data = yaml.safe_load(f)
    elif file_path.suffix.lower() in ['.h5', '.hdf5']:
        # Parse H5 file
        data = parse_h5_topology(input_file)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported formats: .yaml, .yml, .h5, .hdf5")

    # Extract data
    topology = data['topology']
    base_stations = data['base_stations']
    user_equipment = data['user_equipment']

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Group all BSs by unique (x, y) location (all sites)
    loc_to_cids = defaultdict(list)
    loc_to_orientation = dict()
    for bs in base_stations:
        x, y = bs['location']['x'], bs['location']['y']
        loc = (round(x, 6), round(y, 6))  # rounding to avoid float precision issues
        loc_to_cids[loc].append(bs['cid'])
        # Use the orientation of the first sector at this location
        if loc not in loc_to_orientation:
            loc_to_orientation[loc] = bs['antPanelOrientation'][1]
    isd = topology['ISD']
    hex_side = isd / np.sqrt(3)

    # Calculate adaptive marker sizes based on number of UTs
    n_ut = len(user_equipment)
    base_size = 50  # base size for small number of UTs
    min_size = 10   # minimum size to ensure visibility
    max_size = 50   # maximum size for small number of UTs
    
    # Calculate marker size inversely proportional to number of UTs
    if n_ut == 0:
        marker_size = max_size  # Use maximum size when no UTs are present
    else:
        marker_size = max(min_size, min(max_size, base_size * (100 / n_ut)))
    
    # BS marker size should be larger than UE markers
    bs_marker_size = max(30, min(100, marker_size * 1.5))  # 150% of UE marker size, with min 30 and max 100

    # Use blue for all BSs and boundaries
    bs_color = 'tab:blue'

    for loc, cids in loc_to_cids.items():
        x, y = loc
        orientation = loc_to_orientation[loc] * np.pi / 180.0
        
        # Plot BS marker, assuming at least one BS at origin
        ax.scatter(x, y, c=bs_color, marker='^', s=bs_marker_size, label="BS" if x == 0 and y == 0 else "")
        # Draw hexagon (solid blue line)
        hexagon = RegularPolygon((x, y), numVertices=6, radius=hex_side,
                                orientation=np.pi/6, edgecolor=bs_color, facecolor='none', lw=2, zorder=2)
        ax.add_patch(hexagon)
        # Draw sector boundaries (dotted blue lines, 3 per BS, 120 deg apart)
        for i in range(3):
            angle = i * 2 * np.pi / 3 + orientation + np.pi / 3 + np.pi / 6  # Added 30 degrees (π/6) counter-clockwise
            line_length = hex_side / 2 * np.sqrt(3)  # Reduced to 60% of original length
            x_end = x + line_length * np.cos(angle)
            y_end = y + line_length * np.sin(angle)
            ax.plot([x, x_end], [y, y_end], linestyle=':', color=bs_color, lw=1.5, zorder=3)

    # Plot UEs
    ue_x = [ue['location']['x'] for ue in user_equipment]
    ue_y = [ue['location']['y'] for ue in user_equipment]
    ue_outdoor = [ue['outdoor_ind'] for ue in user_equipment]
    
    # Plot outdoor and indoor UEs with different markers
    outdoor_mask = np.array(ue_outdoor) == 1
    indoor_mask = ~outdoor_mask
    
    ax.scatter(np.array(ue_x)[outdoor_mask], np.array(ue_y)[outdoor_mask],
              c='blue', marker='o', s=marker_size, label='Outdoor UE')
    ax.scatter(np.array(ue_x)[indoor_mask], np.array(ue_y)[indoor_mask],
              c='red', marker='s', s=marker_size, label='Indoor UE')

    # Add labels and title
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Network Topology Visualization')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    # Collect all x/y positions from both UEs and BSs
    bs_x = [loc[0] for loc in loc_to_cids.keys()]
    bs_y = [loc[1] for loc in loc_to_cids.keys()]
    all_x = np.array(ue_x + bs_x)
    all_y = np.array(ue_y + bs_y)
    # Account for hexagon radius (hex_side) and a small padding
    padding = hex_side * 0.05
    ax.set_xlim(all_x.min() - hex_side - padding, all_x.max() + hex_side + padding)
    ax.set_ylim(all_y.min() - hex_side - padding, all_y.max() + hex_side + padding)

    # Save the plot
    input_path = Path(input_file)
    output_file = input_path.with_suffix('.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Topology visualization saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Visualize network topology from YAML or H5 file')
    parser.add_argument('input_file', help='Path to the input file containing topology data (.yaml, .yml, .h5, .hdf5)')
    args = parser.parse_args()
    
    try:
        visualize_topology(args.input_file)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == '__main__':
    main() 