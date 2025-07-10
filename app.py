import streamlit as st
import cv2
import numpy as np
from math import atan2, degrees
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

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
    st.title("X-ray Angle Measurement Tool")
    st.write("Upload a full-length lower limb X-ray to measure orthopedic angles")

    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
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
            st.session_state.hip_points = []
            st.session_state.circle_fitting = False

        point_names = list(st.session_state.points.keys())
        current_name = point_names[st.session_state.current_point]

        # Display instructions
        instructions = [
            "1. Hip center (click several points along femoral head, then click 'Fit Circle')",
            "2. Femoral condyles center",
            "3. Most distal medial condyle",
            "4. Most distal lateral condyle",
            "5. Center of medial tibial plateau",
            "6. Center of lateral tibial plateau",
            "7. Center of tibia",
            "8. Center of ankle"
        ]
        
        st.subheader(f"Current Point: {current_name.replace('_', ' ').title()}")
        st.write("Instructions:")
        for i, instr in enumerate(instructions):
            prefix = "➔ " if i == st.session_state.current_point else "○ "
            st.write(prefix + instr)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img_array)
        ax.set_title("Click on the image to mark points")

        # Plot existing points
        for name, point in st.session_state.points.items():
            if point is not None:
                ax.plot(point[0], point[1], 'ro')
                ax.text(point[0], point[1], name.replace('_', ' '), color='yellow')

        # Plot hip circle points if in progress
        if st.session_state.circle_fitting:
            for point in st.session_state.hip_points:
                ax.plot(point[0], point[1], 'ro')

        # Display the plot
        st.pyplot(fig)

        # Point collection controls
        if st.session_state.circle_fitting:
            if st.button('Fit Circle'):
                if len(st.session_state.hip_points) >= 3:
                    points = np.array(st.session_state.hip_points)
                    A = np.column_stack([2*points, np.ones(len(points))])
                    b = (points[:, 0]**2 + points[:, 1]**2)
                    center_x, center_y, _ = np.linalg.lstsq(A, b, rcond=None)[0]
                    st.session_state.points['hip_center'] = (center_x, center_y)
                    st.session_state.current_point += 1
                    st.session_state.circle_fitting = False
                    st.experimental_rerun()
                else:
                    st.warning("Need at least 3 points to fit a circle")
        else:
            if st.button(f'Mark {current_name.replace("_", " ")}'):
                # For hip center, start collecting points for circle fitting
                if current_name == 'hip_center':
                    st.session_state.circle_fitting = True
                    st.experimental_rerun()
                else:
                    # For other points, we'll use Streamlit's file uploader workaround
                    st.warning("On mobile: Long press the image to mark points (see instructions below)")
                    st.info("On desktop: Right-click the image and select 'Save image as', then note the coordinates")

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
            st.subheader("Measurement Results")
            col1, col2 = st.columns(2)
            col1.metric("HKA (Hip-Knee-Ankle Angle)", f"{hka:.1f}°")
            col1.metric("JLCA (Joint Line Congruence Angle)", f"{jlca:.1f}°")
            col2.metric("LDFA (Lateral Distal Femoral Angle)", f"{ldfa:.1f}°")
            col2.metric("MPTA (Medial Proximal Tibial Angle)", f"{mpta:.1f}°")

            # Draw measurement lines
            fig2, ax2 = plt.subplots(figsize=(10, 10))
            ax2.imshow(img_array)
            
            # Plot points
            for name, point in st.session_state.points.items():
                ax2.plot(point[0], point[1], 'ro')
                ax2.text(point[0], point[1], name.replace('_', ' '), color='yellow')

            # Draw lines
            ax2.plot([hc[0], fc[0]], [hc[1], fc[1]], 'b-', label='Mechanical Axis Femur')
            ax2.plot([fc[0], ac[0]], [fc[1], ac[1]], 'b-', label='Mechanical Axis Tibia')
            ax2.plot([mc[0], lc[0]], [mc[1], lc[1]], 'g-', label='Femoral Condyle Line')
            ax2.plot([mtp[0], ltp[0]], [mtp[1], ltp[1]], 'r-', label='Tibial Plateau Line')
            
            ax2.legend()
            st.pyplot(fig2)

            if st.button('Reset Measurements'):
                for key in st.session_state.points.keys():
                    st.session_state.points[key] = None
                st.session_state.current_point = 0
                st.session_state.hip_points = []
                st.session_state.circle_fitting = False
                st.experimental_rerun()

if __name__ == "__main__":
    main()
