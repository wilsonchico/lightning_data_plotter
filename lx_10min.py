import os, sys
import subprocess
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
from datetime import timedelta
from typing import Tuple
from pyproj import Transformer, CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patches as mpatches
from matplotlib.ticker import FixedLocator



def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller .exe"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Constants
# SHAPEFILE_PATH = r"shapefile\ncrprsd_2020.shp"
SHAPEFILE_PATH = resource_path("shapefile/ncrprsd_2020.shp")
DEFAULT_DATA_DIR = r"\\172.17.20.87\rws_local_storage\Lightning Data Repository\Data\PAGASA Morning Huddle"

def prompt_existing_output_dir() -> str:
    """Prompt the user for an output directory that must already exist."""
    while True:
        user_input = input("\nEnter an EXISTING output directory path (plots will be saved here): ").strip()
        if not user_input:
            print("❌ Directory path cannot be empty. Please try again.")
            continue
        
        output_dir = os.path.normpath(user_input)
        
        if not os.path.isdir(output_dir):
            print(f"❌ Directory does not exist: {output_dir}")
            print("   Please ensure the folder is created and the path is correct.\n")
            continue
        
        if not os.access(output_dir, os.W_OK):
            print(f"❌ Directory is not writable: {output_dir}")
            print("   You may not have permission to save files here.\n")
            continue
        
        return output_dir


def get_coordinate(coord_name: str) -> float:
    """Prompt user for a valid decimal degree coordinate."""
    while True:
        try:
            value = float(input(f"Enter {coord_name} (decimal degrees): ").strip())
            return value
        except ValueError:
            print("❌ Invalid input. Please enter a numeric value (e.g., 14.836248).")


def get_radius_meters() -> float:
    """Prompt user for plot radius in meters (default: 3500 m)."""
    while True:
        user_input = input(
            "\nEnter plot radius around target (meters) [default: 3500]: "
        ).strip()
        
        if not user_input:
            print("📍 Using default radius: 3500 m")
            return 3500.0
        
        try:
            radius = float(user_input)
            if radius <= 0:
                print("❌ Radius must be a positive number (e.g., 1000, 5000).")
                continue
            if radius > 50000:
                confirm = input(
                    f"⚠️  Very large radius ({radius:,.0f} m) may reduce detail or slow plotting.\n"
                    f"   Continue? (y/N): "
                ).strip().lower()
                if confirm not in ('y', 'yes'):
                    continue
            print(f"📍 Plot radius set to: {radius:,.0f} m")
            return radius
        except ValueError:
            print("❌ Invalid input. Please enter a number (e.g., 2000, 5000.5).")

def select_basemap() -> str:
    """Prompt user to select a basemap provider and style."""
    print("\n🌍 Select a basemap:")
    print("  1. Esri World Imagery (satellite)")
    print("  2. Esri World Topo Map")
    print("  3. Esri World Street Map")
    print("  4. OpenStreetMap (Standard)")
    print("  5. OpenTopoMap")
    print("  6. CartoDB Positron (light)")
    print("  7. CartoDB Dark Matter (dark)")
    
    basemap_options = {
        '1': ctx.providers.Esri.WorldImagery,
        '2': ctx.providers.Esri.WorldTopoMap,
        '3': ctx.providers.Esri.WorldStreetMap,
        '4': ctx.providers.OpenStreetMap.Mapnik,
        '5': ctx.providers.OpenTopoMap,
        '6': ctx.providers.CartoDB.Positron,
        '7': ctx.providers.CartoDB.DarkMatter,
    }

    while True:
        choice = input("\nEnter choice (1–7) [default: 1 = Esri Imagery]: ").strip()
        if not choice:
            print("📍 Using default: Esri World Imagery")
            return ctx.providers.Esri.WorldImagery
        if choice in basemap_options:
            selected = basemap_options[choice]
            name_map = {
                '1': 'Esri World Imagery',
                '2': 'Esri World Topo Map',
                '3': 'Esri World Street Map',
                '4': 'OpenStreetMap',
                '5': 'OpenTopoMap',
                '6': 'CartoDB Positron',
                '7': 'CartoDB Dark Matter',
            }
            print(f"📍 Selected basemap: {name_map[choice]}")
            return selected
        else:
            print("❌ Invalid choice. Please enter 1–7.")

def add_basemap(ax, crs, provider, zoom: int = 15) -> None:
    """Add selected basemap to the plot (handles CRS and attribution)."""
    try:
        ctx.add_basemap(
            ax,
            crs=crs,
            source=provider,
            zoom=zoom,
            attribution=False  # suppresses watermark (optional)
        )
    except Exception as e:
        print(f"⚠️ Warning: Could not add basemap ({provider}): {e}")
        # Optionally fall back to no basemap or a simple background
        ax.set_facecolor('#f0f0f0')  # light gray fallback


def prompt_directory(default_dir: str) -> str:
    """Prompt user for directory or use default."""
    print(f"\nEnter the directory containing the lightning CSV file.")
    user_input = input(f"[Default: {default_dir}]\nDirectory (press ENTER for default): ").strip()
    return os.path.normpath(user_input if user_input else default_dir)


def prompt_csv_file(data_dir: str) -> str:
    """Prompt user for a CSV filename that exists in the given directory."""
    while True:
        filename = input("Enter CSV filename (e.g., lx_02Oct2025.csv): ").strip()
        if not filename:
            print("❌ Filename cannot be empty. Please try again.")
            continue

        if not filename.lower().endswith('.csv'):
            print("⚠️  Warning: Filename should end with .csv")

        csv_path = os.path.join(data_dir, filename)

        if not os.path.isfile(csv_path):
            print(f"❌ File not found: {csv_path}")
            print("   Please check the filename and ensure it exists in the directory above.\n")
            continue

        return csv_path


def prompt_start_time() -> pd.Timestamp:
    """Prompt user for a valid start time."""
    print("\nEnter the start time (format: yyyy-mm-dd HH:MM, e.g., 2025-10-02 12:00)")
    user_input = input("Start time: ").strip()
    for fmt in ['%Y-%m-%d %H:%M.%f', '%Y-%m-%d %H:%M', '%Y-%m-%d %H:%M:%S']:
        try:
            return pd.to_datetime(user_input, format=fmt)
        except ValueError:
            continue
    raise ValueError("❌ Invalid time format. Please use yyyy-mm-dd HH:MM")


def prompt_end_time(start_time: pd.Timestamp, lightning: pd.DataFrame) -> pd.Timestamp:
    """
    Prompt user for end time.
    - If user enters nothing: use max time in lightning data.
    - If user enters time: must be > start_time.
    """
    max_data_time = lightning['time_utc'].max()
    # min_data_time = lightning['time_utc'].min()

    # print(f"\n💡 Lightning data range: {min_data_time} → {max_data_time}")
    # print(f"   Start time is: {start_time}")

    while True:
        user_input = input(
            "Enter end time (yyyy-mm-dd HH:MM), or press ENTER to use latest data time: "
        ).strip()

        # ▼▼▼ CASE: User pressed ENTER → use max data time ▼▼▼
        if not user_input:
            print(f"⏭️  Using latest data time: {max_data_time}")
            return max_data_time

        # Parse input
        end_time = None
        for fmt in ['%Y-%m-%d %H:%M.%f', '%Y-%m-%d %H:%M', '%Y-%m-%d %H:%M:%S']:
            try:
                end_time = pd.to_datetime(user_input, format=fmt)
                break
            except ValueError:
                continue

        if end_time is None:
            print("❌ Invalid time format. Please use yyyy-mm-dd HH:MM")
            continue

        if end_time <= start_time:
            print("❌ End time must be after the start time. Please try again.")
            continue

        # Warn if beyond data
        if end_time > max_data_time:
            confirm = input(
                f"⚠️  End time ({end_time}) is beyond latest data ({max_data_time}).\n"
                f"   Proceed anyway? (y/N): "
            ).strip().lower()
            if confirm not in ('y', 'yes'):
                print("↩️  Using latest data time instead.")
                return max_data_time

        return end_time


def get_utm_crs(lon: float, lat: float) -> str:
    """Determine best UTM CRS for given lon/lat (Philippines-friendly fallback)."""
    try:
        # Try new pyproj >=3.5+ signature
        try:
            aoi = AreaOfInterest(
                west_lon_degree=lon - 0.01,
                south_lat_degree=lat - 0.01,
                east_lon_degree=lon + 0.01,
                north_lat_degree=lat + 0.01
            )
        except TypeError:
            # Fallback to old pyproj <3.5 signature
            aoi = AreaOfInterest(
                west_lon_deg=lon - 0.01,
                south_lat_deg=lat - 0.01,
                east_lon_deg=lon + 0.01,
                north_lat_deg=lat + 0.01
            )
        
        utm_crs_list = query_utm_crs_info(area_of_interest=aoi)
        if utm_crs_list:
            return f"EPSG:{utm_crs_list[0].code}"
    except Exception as e:
        print(f"⚠️ UTM detection failed: {e}")
    
    return "EPSG:32651"


def add_north_arrow(ax, x, y, size=100, color='white'):
    """Add a simple north arrow to the plot (in meters for projected CRS)."""
    arrow = mpatches.FancyArrowPatch(
        (x, y), (x, y + size * 0.8),
        mutation_scale=size * 0.1,
        arrowstyle='simple',
        facecolor=color,
        edgecolor='black',
        linewidth=0.5,
        zorder=10
    )
    ax.add_patch(arrow)
    ax.text(x, y + size, 'N', ha='center', va='center',
            fontsize=12, fontweight='bold', color=color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.6))


def main() -> None:

    # ▼▼▼ NEW: Lightning data format requirements — shown FIRST ▼▼▼
    print("=" * 70)
    print("⚡ LIGHTNING DATA FORMAT REQUIREMENTS")
    print("=" * 70)
    print("This script expects a CSV file with the following columns:")
    print()
    print("  • time_utc      → UTC timestamp (e.g., '2025-10-02 12:05:23')")
    print("                  Accepts common formats: yyyy-mm-dd HH:MM, HH:MM:SS, or with .ff")
    print()
    print("  • latitude      → Decimal degrees (e.g., 14.5892); positive = North")
    print("  • longitude     → Decimal degrees (e.g., 121.0535); positive = East")
    print()
    print("  • FlashType     → Integer code:")
    print("        0 = Cloud-to-Ground (CG) strike")
    print("        1 = In-Cloud (IC) discharge")
    print()
    print("🔍 Example CSV content:")
    print("time_utc,latitude,longitude,FlashType")
    print("2025-10-02 12:05:23,14.6012,121.0234,0")
    print("2025-10-02 12:06:17,14.5998,121.0261,1")
    print()
    print("❗ If your data uses different column names (e.g., 'lat', 'lon', 'type', 'timestamp'),")
    print("   please rename or preprocess the CSV accordingly before proceeding.")
    print()
    print("💡 Tip: You can open the CSV in Excel or a text editor to verify the headers.")
    print("=" * 70)
    input("\n✅ Press ENTER to continue and begin plotting setup...\n")
    # ▲▲▲ END REQUIREMENTS NOTICE ▲▲▲

    # Output directory
    output_dir = prompt_existing_output_dir()

    # Lightning CSV
    data_dir = prompt_directory(DEFAULT_DATA_DIR)
    csv_path = prompt_csv_file(data_dir)

    # Load data early to show time range
    print("\n⏳ Loading lightning data...")
    lightning = pd.read_csv(csv_path)
    lightning['time_utc'] = pd.to_datetime(lightning['time_utc'])

    # Show data range
    min_data_time = lightning['time_utc'].min()
    max_data_time = lightning['time_utc'].max()
    print(f"\n📊 Lightning Data Time Range:")
    print(f"   📅 From: {min_data_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"   📅 To:   {max_data_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"   ⚡ Total strikes: {len(lightning):,}")

    boundaries = gpd.read_file(SHAPEFILE_PATH)
    if boundaries.crs != "EPSG:4326":
        boundaries = boundaries.to_crs("EPSG:4326")

    # Target info
    print("\nEnter Coordinates of the Target")
    station_lat = get_coordinate("latitude")
    station_lon = get_coordinate("longitude")
    target_name = input("Enter Target Name: ").strip()

    # ▼▼▼ Pre-check: any lightning within 1 km in the entire dataset (bundled by 10 min) ▼▼▼
    print("\n🔎 Checking for any lightning strikes within 1 km of target across all data...")

    # Determine projected CRS (in meters)
    proj_crs = get_utm_crs(station_lon, station_lat)
    transformer = Transformer.from_crs("EPSG:4326", proj_crs, always_xy=True)

    # Transform coordinates
    station_x, station_y = transformer.transform(station_lon, station_lat)
    lightning_x, lightning_y = transformer.transform(
        lightning['longitude'].values,
        lightning['latitude'].values
    )

    # Compute distances (in meters)
    distances = np.sqrt((lightning_x - station_x)**2 + (lightning_y - station_y)**2)
    lightning['distance_m'] = distances

    # Filter lightning within 1 km
    nearby_all = lightning[lightning['distance_m'] <= 1000].copy()

    if not nearby_all.empty:
        print(f"⚡ ALERT: {len(nearby_all)} lightning strike(s) within 1 km of target detected!")
        print("   Grouped by 10-minute intervals (UTC):")

        # Ensure datetime format
        nearby_all['time_utc'] = pd.to_datetime(nearby_all['time_utc'])

        # Round down to nearest 10 minutes (avoid deprecation warning)
        nearby_all['time_bin'] = nearby_all['time_utc'].dt.floor('10min')

        # Count per 10-minute interval
        grouped = nearby_all.groupby('time_bin').size().reset_index(name='count')

        for _, row in grouped.iterrows():
            start_time = row['time_bin']
            end_time = start_time + pd.Timedelta(minutes=10)
            print(f"   • {start_time:%Y-%m-%d %H:%M} – {end_time:%H:%M} : {row['count']} strike(s)")
    else:
        print("✅ No lightning within 1 km of target in the entire dataset.")

    # ▲▲▲ End overall proximity check ▲▲▲

    # ▼▼▼ NEW: Ask user whether to proceed after proximity alert ▼▼▼
    proceed = input("   \nContinue to plot the data? (Y/n): ").strip().lower()
    if proceed in ('n', 'no'):
        print("\n⏹️  Plotting cancelled by user.")
        print("👋 Exiting program. Goodbye!")
        os.system('cls' if os.name == 'nt' else 'clear')  # clear terminal
        print("⟳ Restarting program...\n")
        return True
    else:
        print("✅ Continuing with plot setup...\n")
    # ▲▲▲ END user confirmation after proximity check ▲▲▲

    # ✅ Get radius in meters (user-defined)
    delta = get_radius_meters()  # e.g., 3500.0

    # ▼▼▼ NEW: Basemap selection ▼▼▼
    basemap_provider = select_basemap()
    # ▲▲▲

    # Time setup
    start_time = prompt_start_time()
    end_time = prompt_end_time(start_time, lightning)
    interval = timedelta(minutes=10)

    # Optional: guard rails
    if start_time < min_data_time:
        print(f"\n⚠️  Start time is before earliest data.")
        confirm = input(f"   Adjust to {min_data_time.strftime('%Y-%m-%d %H:%M')}? (Y/n): ").strip().lower()
        if confirm in ('', 'y', 'yes'):
            start_time = min_data_time

    if end_time > max_data_time:
        print(f"\n⚠️  End time is after latest data.")
        confirm = input(f"   Adjust to {max_data_time.strftime('%Y-%m-%d %H:%M')}? (Y/n): ").strip().lower()
        if confirm in ('', 'y', 'yes'):
            end_time = max_data_time

    # Determine UTM CRS
    proj_crs = get_utm_crs(station_lon, station_lat)
    print(f"📍 Using projected CRS: {proj_crs}")

    # Plotting loop
    current_time = start_time
    while current_time <= end_time:
        next_time = current_time + interval
        mask = (lightning['time_utc'] >= current_time) & (lightning['time_utc'] < next_time)
        subset = lightning[mask]

        if subset.empty:
            print(f"🕒 No data for interval {current_time} → {next_time}")
            current_time = next_time
            continue

        cloud_strikes = subset[subset['FlashType'] == 1]   # In-cloud
        ground_strikes = subset[subset['FlashType'] == 0]  # Cloud-to-ground

        # Reproject boundaries
        boundaries_proj = boundaries.to_crs(proj_crs)

        # Transformers
        transformer = Transformer.from_crs("EPSG:4326", proj_crs, always_xy=True)
        inv_transformer = Transformer.from_crs(proj_crs, "EPSG:4326", always_xy=True)

        # Transform station
        station_x, station_y = transformer.transform(station_lon, station_lat)

        # Set plot bounds (user-defined delta in meters)
        xlim_min, xlim_max = station_x - delta, station_x + delta
        ylim_min, ylim_max = station_y - delta, station_y + delta

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot boundaries
        boundaries_proj.boundary.plot(ax=ax, color='white', linewidth=1, zorder=2)

        # Plot strikes
        if not ground_strikes.empty:
            gx, gy = transformer.transform(
                ground_strikes['longitude'].values,
                ground_strikes['latitude'].values
            )
            ax.scatter(gx, gy, color='yellow', edgecolor='black', s=50,
                       label='Cloud to Ground', zorder=5)

        if not cloud_strikes.empty:
            cx, cy = transformer.transform(
                cloud_strikes['longitude'].values,
                cloud_strikes['latitude'].values
            )
            ax.scatter(cx, cy, color='magenta', edgecolor='black', s=50,
                       label='In Cloud', zorder=5)

        # Plot target
        ax.scatter(station_x, station_y, marker='^', color='red', s=120,
                   zorder=7, label='Target Location', edgecolor='black', linewidth=0.8)
        ax.text(station_x + delta * 0.03, station_y, target_name,
                fontsize=10, color='black',
                verticalalignment='center',
                bbox=dict(boxstyle="square",facecolor='white',edgecolor='none',alpha=0.3))

        # Set limits
        ax.set_xlim(xlim_min, xlim_max)
        ax.set_ylim(ylim_min, ylim_max)

        # Basemap
        add_basemap(ax, crs=proj_crs, provider=basemap_provider, zoom=15)

        # Scale bar
        scalebar = ScaleBar(
            dx=1, units='m',
            location='lower right',
            frameon=True,
            color='black',
            box_alpha=0.8,
            scale_loc='top',
            label_loc='bottom',
            font_properties={'size': 10}
        )
        ax.add_artist(scalebar)

        # North arrow (top-left)
        arrow_x = xlim_min + delta * 0.15
        arrow_y = ylim_max - delta * 0.2
        add_north_arrow(ax, arrow_x, arrow_y, size=delta * 0.04, color='white')

        # ▼▼▼ Axis labels as Longitude/Latitude (°), NO WARNINGS ▼▼▼
        plt.draw()  # force tick computation
        xticks_m = ax.get_xticks()
        yticks_m = ax.get_yticks()

        try:
            xticks_lon, _ = inv_transformer.transform(xticks_m, np.full_like(xticks_m, station_y))
            _, yticks_lat = inv_transformer.transform(np.full_like(yticks_m, station_x), yticks_m)

            x_labels = [f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}" for lon in xticks_lon]
            y_labels = [f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}" for lat in yticks_lat]

            ax.xaxis.set_major_locator(FixedLocator(xticks_m))
            ax.yaxis.set_major_locator(FixedLocator(yticks_m))
            ax.set_xticklabels(x_labels)
            ax.set_yticklabels(y_labels)

            ax.set_xlabel('Longitude', fontsize=11, color='black')
            ax.set_ylabel('Latitude', fontsize=11, color='black')

        except Exception as e:
            print(f"⚠️ Tick conversion failed: {e}")
            ax.set_xlabel('Easting (m)', fontsize=11, color='black')
            ax.set_ylabel('Northing (m)', fontsize=11, color='black')
        # ▲▲▲

        # Title & styling
        title = (f"Lightning Strikes | {current_time.strftime('%Y-%m-%d %H:%M')} "
                 f"→ {next_time.strftime('%Y-%m-%d %H:%M')} UTC")
        ax.set_title(title, fontsize=12, color='black')

        ax.tick_params(colors='black')
        ax.spines[:].set_color('black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.grid(True, linestyle='--', alpha=0.4, color='white')

        # Legend
        legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_alpha(0.7)
        for text in legend.get_texts():
            text.set_color('black')

        # Save
        filename_out = f"lightning_plot_{current_time.strftime('%Y%m%d_%H%M')}.png"
        filepath = os.path.join(output_dir, filename_out)
        plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"✅ Saved: {filename_out}")

        current_time = next_time

    print("\n🎉 All plots generated successfully!")

    again = input("📊 Do you want to plot again? (Y/N): ").strip().lower()

    if again != 'y':
        print("\n👋 Exiting program. Goodbye!")
        return False  # signal to end program
    else:
        os.system('cls' if os.name == 'nt' else 'clear')  # clear terminal
        print("⟳ Restarting program...\n")
        return True   # signal to restart


# --- Program entry point ---
if __name__ == "__main__":
    while True:
        restart = main()
        if not restart:
            break