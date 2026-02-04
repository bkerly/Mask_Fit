"""
NIOSH Respirator Mask Fitting System
A professional application for face scanning, mask recommendation, and fit testing
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors
import tempfile

# Page configuration
st.set_page_config(
    page_title="NIOSH Mask Fitting System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Force light mode */
    .stApp {
        background-color: #ffffff;
        color: #262730;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    .step-header {
        font-size: 1.8rem;
        color: #2ca02c;
        margin-top: 1rem;
        font-weight: 600;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
        color: #262730;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        color: #856404;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        color: #155724;
        margin: 1rem 0;
    }
    
    /* Ensure all text is readable */
    .info-box *, .warning-box *, .success-box * {
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'measurements' not in st.session_state:
    st.session_state.measurements = None
if 'recommendation' not in st.session_state:
    st.session_state.recommendation = None
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'subject_name' not in st.session_state:
    st.session_state.subject_name = ""
if 'subject_dob' not in st.session_state:
    st.session_state.subject_dob = None
if 'available_masks' not in st.session_state:
    st.session_state.available_masks = []

# NIOSH headform reference data
NIOSH_HEADFORMS = {
    'small': {
        'bizygomatic_breadth': (125, 135),
        'menton_sellion': (105, 115),
        'description': 'Small face size',
        'percentile': '~10-25th percentile'
    },
    'medium': {
        'bizygomatic_breadth': (135, 145),
        'menton_sellion': (115, 125),
        'description': 'Medium face size',
        'percentile': '~25-75th percentile'
    },
    'large': {
        'bizygomatic_breadth': (145, 160),
        'menton_sellion': (125, 135),
        'description': 'Large face size',
        'percentile': '~75-95th percentile'
    },
    'long_narrow': {
        'bizygomatic_breadth': (125, 140),
        'menton_sellion': (125, 140),
        'description': 'Long and narrow face',
        'percentile': 'Specialized fit - longer, narrower features'
    },
    'short_wide': {
        'bizygomatic_breadth': (145, 165),
        'menton_sellion': (105, 120),
        'description': 'Short and wide face',
        'percentile': 'Specialized fit - shorter, wider features'
    }
}

# Complete mask database
ALL_MASKS = {
    'small': [
        {'brand': '3M', 'model': '8210 N95', 'size': 'Small', 'fit_score': 95},
        {'brand': 'Honeywell', 'model': 'DF300N95', 'size': 'Small', 'fit_score': 92},
        {'brand': 'Moldex', 'model': '2200 N95', 'size': 'Small', 'fit_score': 90},
    ],
    'medium': [
        {'brand': '3M', 'model': '8210 N95', 'size': 'Regular', 'fit_score': 96},
        {'brand': 'Moldex', 'model': '2200 N95', 'size': 'Medium/Large', 'fit_score': 94},
        {'brand': 'Honeywell', 'model': 'DF300N95', 'size': 'Regular', 'fit_score': 93},
        {'brand': 'MSA', 'model': 'Advantage 200', 'size': 'Medium', 'fit_score': 92},
    ],
    'large': [
        {'brand': '3M', 'model': '8210Plus N95', 'size': 'Large', 'fit_score': 95},
        {'brand': 'Moldex', 'model': '2200 N95', 'size': 'Large', 'fit_score': 93},
        {'brand': 'MSA', 'model': 'Advantage 200', 'size': 'Large', 'fit_score': 92},
    ],
    'long_narrow': [
        {'brand': '3M', 'model': '9205+ Aura', 'size': 'Regular', 'fit_score': 94},
        {'brand': 'Moldex', 'model': '2200 N95', 'size': 'Medium/Large', 'fit_score': 91},
        {'brand': '3M', 'model': '8210 N95', 'size': 'Regular', 'fit_score': 89},
    ],
    'short_wide': [
        {'brand': 'MSA', 'model': 'Advantage 200', 'size': 'Large', 'fit_score': 93},
        {'brand': '3M', 'model': '8210Plus N95', 'size': 'Large', 'fit_score': 92},
        {'brand': 'Moldex', 'model': '2200 N95', 'size': 'Large', 'fit_score': 90},
    ]
}

class FaceMeasurement:
    """Class to handle face measurement using MediaPipe"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def process_image(self, image):
        """Process image and extract facial measurements"""
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                return None, None
            
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = image.shape
            
            # Calculate measurements
            measurements = self._calculate_measurements(landmarks, w, h)
            
            # Draw landmarks on image for visualization
            annotated_image = image.copy()
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=results.multi_face_landmarks[0],
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            return measurements, annotated_image
    
    def _calculate_distance(self, point1, point2, w, h):
        """Calculate distance between two landmarks in mm (estimated)"""
        x1, y1 = point1.x * w, point1.y * h
        x2, y2 = point2.x * w, point2.y * h
        pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Calibration factor
        mm_per_pixel = 140 / 180
        
        return pixel_distance * mm_per_pixel
    
    def _calculate_measurements(self, landmarks, w, h):
        """Extract key facial measurements"""
        measurements = {
            'bizygomatic_breadth': self._calculate_distance(landmarks[454], landmarks[234], w, h),
            'menton_sellion': self._calculate_distance(landmarks[152], landmarks[6], w, h),
            'face_width': self._calculate_distance(landmarks[454], landmarks[234], w, h),
            'face_length': self._calculate_distance(landmarks[152], landmarks[10], w, h),
        }
        
        return measurements

class MaskRecommender:
    """Class to recommend masks based on measurements"""
    
    def categorize_face(self, measurements):
        """Categorize face into NIOSH panel categories"""
        bizyg = measurements['bizygomatic_breadth']
        mensell = measurements['menton_sellion']
        
        aspect_ratio = mensell / bizyg if bizyg > 0 else 1.0
        
        category = None
        confidence = 0
        
        for cat_name, cat_data in NIOSH_HEADFORMS.items():
            bizyg_min, bizyg_max = cat_data['bizygomatic_breadth']
            mensell_min, mensell_max = cat_data['menton_sellion']
            
            bizyg_match = bizyg_min <= bizyg <= bizyg_max
            mensell_match = mensell_min <= mensell <= mensell_max
            
            if bizyg_match and mensell_match:
                category = cat_name
                bizyg_center = (bizyg - bizyg_min) / (bizyg_max - bizyg_min)
                mensell_center = (mensell - mensell_min) / (mensell_max - mensell_min)
                confidence = 100 - abs(0.5 - bizyg_center) * 100 - abs(0.5 - mensell_center) * 100
                break
        
        if not category:
            if aspect_ratio > 1.0:
                category = 'long_narrow'
                confidence = 75
            elif aspect_ratio < 0.85:
                category = 'short_wide'
                confidence = 75
            elif bizyg < 135:
                category = 'small'
                confidence = 70
            elif bizyg > 145:
                category = 'large'
                confidence = 70
            else:
                category = 'medium'
                confidence = 80
        
        return category, max(min(confidence, 99), 60)
    
    def get_recommendations(self, category, available_masks=None):
        """Get mask recommendations for a category"""
        all_category_masks = ALL_MASKS.get(category, ALL_MASKS['medium'])
        
        if not available_masks:
            return all_category_masks
        
        # Filter to only available masks
        filtered = []
        for mask in all_category_masks:
            mask_id = f"{mask['brand']} {mask['model']} - {mask['size']}"
            if mask_id in available_masks:
                filtered.append(mask)
        
        return filtered if filtered else all_category_masks

def generate_pdf_report(subject_name, subject_dob, measurements, recommendation, annotated_image):
    """Generate PDF report with all fitting information"""
    
    # Create temporary file
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2ca02c'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("NIOSH RESPIRATOR FITTING REPORT", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Subject Information
    story.append(Paragraph("Subject Information", heading_style))
    subject_data = [
        ['Name:', subject_name],
        ['Date of Birth:', subject_dob.strftime('%B %d, %Y') if subject_dob else 'Not provided'],
        ['Test Date:', datetime.now().strftime('%B %d, %Y')],
        ['Test Time:', datetime.now().strftime('%I:%M %p')]
    ]
    subject_table = Table(subject_data, colWidths=[2*inch, 4*inch])
    subject_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f4f8')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(subject_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Facial Measurements
    story.append(Paragraph("Facial Measurements", heading_style))
    meas_data = [
        ['Measurement', 'Value (mm)'],
        ['Bizygomatic Breadth', f"{measurements['bizygomatic_breadth']:.1f}"],
        ['Menton-Sellion Length', f"{measurements['menton_sellion']:.1f}"],
        ['Face Width', f"{measurements['face_width']:.1f}"],
        ['Face Length', f"{measurements['face_length']:.1f}"]
    ]
    meas_table = Table(meas_data, colWidths=[3*inch, 2*inch])
    meas_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(meas_table)
    story.append(Spacer(1, 0.3*inch))
    
    # NIOSH Category
    story.append(Paragraph("NIOSH Face Size Category", heading_style))
    category = recommendation['category']
    category_data = [
        ['Category:', category.replace('_', ' ').title()],
        ['Description:', NIOSH_HEADFORMS[category]['description']],
        ['Population:', NIOSH_HEADFORMS[category]['percentile']],
        ['Match Confidence:', f"{recommendation['confidence']:.0f}%"]
    ]
    category_table = Table(category_data, colWidths=[2*inch, 4*inch])
    category_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#d4edda')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    story.append(category_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Recommended Respirators
    story.append(Paragraph("Recommended Respirators", heading_style))
    rec_data = [['#', 'Brand', 'Model', 'Size', 'Expected Fit']]
    for i, mask in enumerate(recommendation['masks'], 1):
        rec_data.append([
            str(i),
            mask['brand'],
            mask['model'],
            mask['size'],
            f"{mask['fit_score']}%"
        ])
    rec_table = Table(rec_data, colWidths=[0.5*inch, 1.5*inch, 2*inch, 1.5*inch, 1*inch])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Face scan image
    if annotated_image is not None:
        story.append(Paragraph("Facial Scan with Landmark Detection", heading_style))
        
        # Convert image to temporary file
        img_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        img_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.save(img_temp.name)
        
        # Add to PDF (scaled to fit)
        img_reportlab = RLImage(img_temp.name, width=4*inch, height=3*inch)
        story.append(img_reportlab)
        story.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    story.append(Paragraph("Important Information", heading_style))
    disclaimer_text = """
    <para>
    <b>NEXT STEPS:</b><br/>
    1. Obtain one of the recommended respirators<br/>
    2. Perform user seal check before each use<br/>
    3. Consider quantitative fit testing if required by your employer<br/>
    4. Follow all manufacturer and OSHA guidelines<br/><br/>
    
    <b>DISCLAIMER:</b><br/>
    This report is based on automated facial measurements and should be used as a preliminary 
    guide only. Proper fit testing by a qualified professional may be required for workplace use. 
    Always follow OSHA regulations (29 CFR 1910.134) and manufacturer guidelines. This screening 
    does not replace professional fit testing or certification for use in hazardous environments.
    </para>
    """
    story.append(Paragraph(disclaimer_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    
    return pdf_buffer

def main():
    # Header
    st.markdown('<h1 class="main-header">NIOSH Respirator Mask Fitting System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Subject Information")
        st.session_state.subject_name = st.text_input("Subject Name", value=st.session_state.subject_name)
        st.session_state.subject_dob = st.date_input(
            "Date of Birth",
            value=st.session_state.subject_dob,
            max_value=datetime.now().date()
        )
        
        st.markdown("---")
        st.markdown("### Available Masks")
        st.markdown("Select masks you have on hand:")
        
        # Create list of all unique masks
        all_mask_options = []
        for category_masks in ALL_MASKS.values():
            for mask in category_masks:
                mask_id = f"{mask['brand']} {mask['model']} - {mask['size']}"
                if mask_id not in all_mask_options:
                    all_mask_options.append(mask_id)
        
        all_mask_options.sort()
        
        st.session_state.available_masks = st.multiselect(
            "Filter results to these masks:",
            options=all_mask_options,
            default=st.session_state.available_masks,
            help="Leave empty to show all recommendations"
        )
        
        st.markdown("---")
        st.markdown("### Navigation")
        
        steps = {
            1: "Face Scan",
            2: "Analysis & Recommendation",
            3: "Fit Test Protocol"
        }
        
        for step_num, step_name in steps.items():
            if st.button(step_name, key=f"nav_{step_num}", use_container_width=True):
                st.session_state.current_step = step_num
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This system uses NIOSH anthropometric data to recommend
        properly fitting respirators based on facial measurements.
        
        **Data Source:** NIOSH 2003 Anthropometric Survey (3,997 subjects)
        """)
        
        if st.button("Start Over", use_container_width=True):
            # Keep subject info and available masks
            name = st.session_state.subject_name
            dob = st.session_state.subject_dob
            masks = st.session_state.available_masks
            
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            st.session_state.current_step = 1
            st.session_state.subject_name = name
            st.session_state.subject_dob = dob
            st.session_state.available_masks = masks
            st.rerun()
    
    # Main content area
    if st.session_state.current_step == 1:
        show_face_scan()
    elif st.session_state.current_step == 2:
        show_analysis()
    elif st.session_state.current_step == 3:
        show_fit_test()

def show_face_scan():
    """Step 1: Face scanning interface"""
    st.markdown('<h2 class="step-header">Step 1: Face Scan</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Instructions:</strong>
    <ul>
        <li>Position your face directly in front of the camera</li>
        <li>Ensure good lighting</li>
        <li>Remove glasses and face coverings</li>
        <li>Maintain a neutral expression</li>
        <li>Face the camera straight on</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Camera input
        img_file_buffer = st.camera_input("Capture your face")
        
        if img_file_buffer is not None:
            # Read image
            bytes_data = img_file_buffer.getvalue()
            image = Image.open(io.BytesIO(bytes_data))
            image_np = np.array(image)
            
            # Process image
            with st.spinner("Analyzing facial features..."):
                face_measurer = FaceMeasurement()
                measurements, annotated_image = face_measurer.process_image(image_np)
            
            if measurements:
                st.session_state.measurements = measurements
                st.session_state.captured_image = annotated_image
                st.success("Face detected and analyzed successfully")
                
                # Show annotated image - FIXED: use_container_width instead of use_column_width
                st.image(annotated_image, caption="Detected Facial Landmarks", use_container_width=True)
                
                if st.button("Continue to Analysis", type="primary", use_container_width=True):
                    st.session_state.current_step = 2
                    st.rerun()
            else:
                st.error("No face detected. Please try again with better lighting and positioning.")
    
    with col2:
        st.markdown("### Tips for Best Results")
        st.markdown("""
        - **Distance**: Sit about 18-24 inches from camera
        - **Lighting**: Face a window or light source
        - **Background**: Use a plain background if possible
        - **Hair**: Pull hair back from face
        - **Expression**: Keep face relaxed and neutral
        """)
        
        if st.session_state.measurements:
            st.markdown("### Detected Measurements")
            measurements = st.session_state.measurements
            st.metric("Face Width", f"{measurements['bizygomatic_breadth']:.1f} mm")
            st.metric("Face Length", f"{measurements['menton_sellion']:.1f} mm")

def show_analysis():
    """Step 2: Analysis and mask recommendation"""
    st.markdown('<h2 class="step-header">Step 2: Analysis & Recommendation</h2>', unsafe_allow_html=True)
    
    if not st.session_state.measurements:
        st.warning("Please complete face scan first.")
        if st.button("Go to Face Scan"):
            st.session_state.current_step = 1
            st.rerun()
        return
    
    measurements = st.session_state.measurements
    recommender = MaskRecommender()
    category, confidence = recommender.categorize_face(measurements)
    recommendations = recommender.get_recommendations(category, st.session_state.available_masks)
    
    st.session_state.recommendation = {
        'category': category,
        'confidence': confidence,
        'masks': recommendations
    }
    
    # Display results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Your Facial Measurements")
        
        # Measurements table
        meas_data = {
            'Measurement': [
                'Bizygomatic Breadth',
                'Menton-Sellion Length',
                'Face Width',
                'Face Length'
            ],
            'Value (mm)': [
                f"{measurements['bizygomatic_breadth']:.1f}",
                f"{measurements['menton_sellion']:.1f}",
                f"{measurements['face_width']:.1f}",
                f"{measurements['face_length']:.1f}"
            ]
        }
        st.table(pd.DataFrame(meas_data))
        
        # Category information
        st.markdown(f"""
        <div class="success-box">
        <h4>NIOSH Face Size Category: {category.replace('_', ' ').title()}</h4>
        <p><strong>Description:</strong> {NIOSH_HEADFORMS[category]['description']}</p>
        <p><strong>Population:</strong> {NIOSH_HEADFORMS[category]['percentile']}</p>
        <p><strong>Match Confidence:</strong> {confidence:.0f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Recommended Respirators")
        
        if not recommendations:
            st.warning("No masks match your available inventory. Showing all recommendations for your category.")
            recommendations = recommender.get_recommendations(category, None)
        
        for i, mask in enumerate(recommendations, 1):
            st.markdown(f"""
            <div class="info-box">
            <h4>#{i} - {mask['brand']} {mask['model']}</h4>
            <p><strong>Size:</strong> {mask['size']}</p>
            <p><strong>Expected Fit Score:</strong> {mask['fit_score']}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        <strong>Important:</strong> These are preliminary recommendations based on facial measurements.
        Always perform a proper fit test before using any respirator in a hazardous environment.
        </div>
        """, unsafe_allow_html=True)
    
    # Show comparison with NIOSH headform
    st.markdown("### Comparison with NIOSH Reference Headform")
    
    headform_data = NIOSH_HEADFORMS[category]
    comparison_data = {
        'Measurement': ['Bizygomatic Breadth', 'Menton-Sellion Length'],
        'Your Value': [
            f"{measurements['bizygomatic_breadth']:.1f} mm",
            f"{measurements['menton_sellion']:.1f} mm"
        ],
        'Reference Range': [
            f"{headform_data['bizygomatic_breadth'][0]}-{headform_data['bizygomatic_breadth'][1]} mm",
            f"{headform_data['menton_sellion'][0]}-{headform_data['menton_sellion'][1]} mm"
        ]
    }
    st.table(pd.DataFrame(comparison_data))
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Face Scan", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()
    with col2:
        if st.button("Continue to Fit Test", type="primary", use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()

def show_fit_test():
    """Step 3: User seal check/fit test protocol"""
    st.markdown('<h2 class="step-header">Step 3: Respirator Fit Test Protocol</h2>', unsafe_allow_html=True)
    
    if not st.session_state.recommendation:
        st.warning("Please complete analysis first.")
        if st.button("Go to Analysis"):
            st.session_state.current_step = 2
            st.rerun()
        return
    
    st.markdown("""
    <div class="info-box">
    <h3>NIOSH/OSHA User Seal Check Procedure</h3>
    <p>Before each use of your respirator, you must perform a user seal check to ensure proper fit.
    Follow these steps carefully:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different fit test components
    tab1, tab2, tab3 = st.tabs(["Pre-Check", "Positive Pressure Test", "Negative Pressure Test"])
    
    with tab1:
        st.markdown("### Pre-Donning Checklist")
        
        checklist_items = [
            "Respirator is clean and in good condition",
            "Straps are not twisted or damaged",
            "Exhalation valve (if present) moves freely",
            "No visible cracks or holes in mask material",
            "Your face is clean-shaven (for tight-fitting respirators)",
            "Hair is pulled back and not interfering with seal"
        ]
        
        st.markdown("Check each item before proceeding:")
        completed = []
        for item in checklist_items:
            completed.append(st.checkbox(item))
        
        if all(completed):
            st.success("Pre-check complete! Proceed to seal check tests.")
        else:
            st.warning("Complete all checklist items before proceeding.")
    
    with tab2:
        st.markdown("### Positive Pressure Test")
        
        st.markdown("""
        **Purpose:** Verify that the facepiece seals properly against your face.
        
        **Procedure:**
        1. **Don the respirator** according to manufacturer instructions
        2. **Cover the exhalation valve** completely with your palm or a thin material
        3. **Exhale gently** into the mask
        4. **Hold your breath** for 10 seconds
        
        **Expected Result:**
        - The facepiece should **bulge slightly outward**
        - You should feel a **slight positive pressure** inside the mask
        - **No air should leak out** at the edges
        
        **If you fail:**
        - Readjust the straps and try again
        - Check for hair or clothing interfering with seal
        - Try a different size or model if problems persist
        """)
        
        test_result = st.radio(
            "Test Result:",
            ["Not performed", "Pass - No leaks detected", "Fail - Air leaked"],
            key="positive_test"
        )
        
        if test_result == "Pass - No leaks detected":
            st.success("Positive pressure test passed!")
        elif test_result == "Fail - Air leaked":
            st.error("Test failed. Readjust and retry, or try different size/model.")
    
    with tab3:
        st.markdown("### Negative Pressure Test")
        
        st.markdown("""
        **Purpose:** Verify that the facepiece seals properly against your face.
        
        **Procedure:**
        1. **Keep the respirator on** from the previous test
        2. **Cover the inlet openings** with your palms (or filter surfaces for N95)
        3. **Inhale gently**
        4. **Hold your breath** for 10 seconds
        
        **Expected Result:**
        - The facepiece should **collapse slightly inward**
        - The mask should remain **collapsed** while holding breath
        - **No air should leak in** at the edges
        
        **If you fail:**
        - Readjust the straps and try again
        - Check seal around nose bridge (common leak point)
        - Ensure proper placement on face
        - Try a different size or model if problems persist
        """)
        
        test_result = st.radio(
            "Test Result:",
            ["Not performed", "Pass - No leaks detected", "Fail - Air leaked"],
            key="negative_test"
        )
        
        if test_result == "Pass - No leaks detected":
            st.success("Negative pressure test passed!")
        elif test_result == "Fail - Air leaked":
            st.error("Test failed. Readjust and retry, or try different size/model.")
    
    # Final recommendations
    st.markdown("---")
    st.markdown("### Final Recommendations")
    
    rec = st.session_state.recommendation
    
    st.markdown(f"""
    <div class="success-box">
    <h4>Your Recommended Respirators ({rec['category'].replace('_', ' ').title()})</h4>
    """, unsafe_allow_html=True)
    
    for mask in rec['masks']:
        st.markdown(f"- **{mask['brand']} {mask['model']}** - Size: {mask['size']}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
    <h4>Important Safety Information</h4>
    <ul>
        <li><strong>Perform user seal check EVERY TIME</strong> you don the respirator</li>
        <li><strong>Quantitative fit testing</strong> may be required by your employer (OSHA 1910.134)</li>
        <li><strong>Replace respirator</strong> if damaged, breathing becomes difficult, or contaminated</li>
        <li><strong>Do not use</strong> in IDLH (Immediately Dangerous to Life or Health) atmospheres</li>
        <li><strong>Consult safety professional</strong> for workplace-specific requirements</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Download PDF report
    st.markdown("### Download Report")
    
    if not st.session_state.subject_name:
        st.warning("Please enter subject name in the sidebar before generating report.")
    else:
        if st.button("Generate PDF Report", type="primary"):
            with st.spinner("Generating PDF report..."):
                pdf_buffer = generate_pdf_report(
                    st.session_state.subject_name,
                    st.session_state.subject_dob,
                    st.session_state.measurements,
                    st.session_state.recommendation,
                    st.session_state.captured_image
                )
            
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name=f"respirator_fitting_{st.session_state.subject_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Analysis", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()
    with col2:
        if st.button("Start New Fitting", use_container_width=True):
            # Keep subject info and available masks
            name = st.session_state.subject_name
            dob = st.session_state.subject_dob
            masks = st.session_state.available_masks
            
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            st.session_state.current_step = 1
            st.session_state.subject_name = name
            st.session_state.subject_dob = dob
            st.session_state.available_masks = masks
            st.rerun()

if __name__ == "__main__":
    main()
