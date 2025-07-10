import streamlit as st
import cv2
import numpy as np
from math import atan2, degrees
from PIL import Image
import matplotlib.pyplot as plt

def angle_between_lines(a1, a2, b1, b2):
    v1 = a2 - a1
    v2 = b2 - b1
    angle_rad = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    angle_deg = np.degrees(angle_rad)
    angle_deg = np.abs(angle_deg)
    if angle_deg > 180:
        angle_deg = 360 - angle_deg
    return angle_deg

def main():
    st.title("📐 X-ray Angle Measurement Tool")
    st.write("Upload a full-length lower limb X-ray to measure orthopedic angles")

    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Add image orientation controls
        col1, col2, col3 = st.columns(3)
        with col1:
            flip_horizontal = st.checkbox("Flip Horizontal (Mirror)")
        with col2:
            flip_vertical = st.checkbox("Flip Vertical (Upside Down)")
        with col3:
            rotate_90 = st.checkbox("Rotate 90°")
        
        # Apply transformations
        if flip_horizontal:
            img_array = np.fliplr(img_array)
        if flip_vertical:
            img_array = np.flipud(img_array)
        if rotate_90:
            img_array = np.rot90(img_array)
        
        height, width = img_array.shape[0], img_array.shape[1]
        
        if 'points' not in st.session_state:
            st.session_state.points = {
                'hip_center': None,
                'femoral_condyles_center': None,
                'medial_condyle': None,
                'lateral_condyle': None,
                'medial_tibial_plateau': None,
                'lateral_tibial_plateau': None,
                'tibia_center': None,
                'ankle_center': None
            }
            st.session_state.current_point = 0
            st.session_state.original_image = img_array.copy()

        point_names = list(st.session_state.points.keys())
        current_name = point_names[st.session_state.current_point]

        # Display image with current markings
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img_array)
        
        # Plot existing points
        for name, point in st.session_state.points.items():
            if point is not None:
                # Transform points if image was flipped
                x, y = point[0], point[1]
                if flip_horizontal:
                    x = width - x - 1
                if flip_vertical:
                    y = height - y - 1
                if rotate_90:
                    x, y = y, width - x - 1
                ax.plot(x, y, 'ro')
                ax.text(x, y, name.replace('_', ' '), color='yellow')
        
        st.pyplot(fig)

        # Coordinate input
        st.subheader(f"Mark {current_name.replace('_', ' ').title()}")
        
        col1, col2 = st.columns(2)
        x = col1.number_input(f"X coordinate (0-{width})", 
                            min_value=0, max_value=width, 
                            value=width//2, key=f"{current_name}_x")
        y = col2.number_input(f"Y coordinate (0-{height})", 
                             min_value=0, max_value=height, 
                             value=height//2, key=f"{current_name}_y")

        if st.button(f"Save {current_name.replace('_', ' ')}"):
            # Save original coordinates (before flipping)
            if flip_horizontal:
                x = width - x - 1
            if flip_vertical:
                y = height - y - 1
            if rotate_90:
                x, y = height - y - 1, x
            st.session_state.points[current_name] = (x, y)
            st.session_state.current_point += 1
            st.experimental_rerun()

        # Special handling for hip center (circle fitting)
        if current_name == 'hip_center':
            st.warning("For hip center: Mark 3+ points around femoral head circumference")
            if st.button("Add Hip Point"):
                if 'hip_points' not in st.session_state:
                    st.session_state.hip_points = []
                # Save original coordinates
                x_orig, y_orig = x, y
                if flip_horizontal:
                    x_orig = width - x_orig - 1
                if flip_vertical:
                    y_orig = height - y_orig - 1
                if rotate_90:
                    x_orig, y_orig = height - y_orig - 1, x_orig
                st.session_state.hip_points.append((x_orig, y_orig))
                st.experimental_rerun()
                
            if 'hip_points' in st.session_state and len(st.session_state.hip_points) >= 3:
                if st.button("Calculate Hip Center"):
                    points = np.array(st.session_state.hip_points)
                    A = np.column_stack([2*points, np.ones(len(points))])
                    b = (points[:, 0]**2 + points[:, 1]**2)
                    center_x, center_y, _ = np.linalg.lstsq(A, b, rcond=None)[0]
                    st.session_state.points['hip_center'] = (center_x, center_y)
                    st.session_state.current_point += 1
                    st.experimental_rerun()

        # Calculate angles when all points are marked
        if all(p is not None for p in st.session_state.points.values()):
            hc = np.array(st.session_state.points['hip_center'])
            fc = np.array(st.session_state.points['femoral_condyles_center'])
            mc = np.array(st.session_state.points['medial_condyle'])
            lc = np.array(st.session_state.points['lateral_condyle'])
            mtp = np.array(st.session_state.points['medial_tibial_plateau'])
            ltp = np.array(st.session_state.points['lateral_tibial_plateau'])
            tc = np.array(st.session_state.points['tibia_center'])
            ac = np.array(st.session_state.points['ankle_center'])

            # Calculate angles
            hka = angle_between_lines(hc, fc, ac, fc)
            jlca = angle_between_lines(mc, lc, mtp, ltp)
            ldafa = angle_between_lines(hc, fc, mc, lc)
            mpta = angle_between_lines(ac, tc, mtp, ltp)

            # Display results
            st.success("Measurement Complete!")
            col1, col2 = st.columns(2)
            col1.metric("HKA (Hip-Knee-Ankle)", f"{hka:.1f}°")
            col1.metric("JLCA (Joint Line Congruence)", f"{jlca:.1f}°")
            col2.metric("LDFA (Lateral Distal Femoral)", f"{ldafa:.1f}°")
            col2.metric("MPTA (Medial Proximal Tibial)", f"{mpta:.1f}°")

            # Draw final plot (on original image)
            fig2, ax2 = plt.subplots(figsize=(10, 10))
            ax2.imshow(st.session_state.original_image)
            
            # Plot all points
            for name, point in st.session_state.points.items():
                ax2.plot(point[0], point[1], 'ro')
                ax2.text(point[0], point[1], name.replace('_', ' '), color='yellow')
            
            # Draw measurement lines
            ax2.plot([hc[0], fc[0]], [hc[1], fc[1]], 'b-', label='Mechanical Axis Femur')
            ax2.plot([fc[0], ac[0]], [fc[1], ac[1]], 'b-', label='Mechanical Axis Tibia')
            ax2.plot([mc[0], lc[0]], [mc[1], lc[1]], 'g-', label='Femoral Condyle Line')
            ax2.plot([mtp[0], ltp[0]], [mtp[1], ltp[1]], 'r-', label='Tibial Plateau Line')
            ax2.legend()
            st.pyplot(fig2)

            if st.button("Start New Measurement"):
                st.session_state.clear()
                st.experimental_rerun()

if __name__ == "__main__":
    main()
