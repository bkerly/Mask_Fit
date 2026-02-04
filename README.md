# NIOSH Respirator Mask Fitting System

A laptop-based prototype system for facial scanning, mask size recommendation, and standardized fit testing based on NIOSH anthropometric data.

## Overview

This application provides a complete workflow for respirator fitting:
1. **Face Scanning** - Uses webcam and MediaPipe to capture facial measurements
2. **Mask Recommendation** - Matches measurements to NIOSH size categories and recommends specific masks
3. **Fit Test Protocol** - Guides users through standardized OSHA/NIOSH seal check procedures

## Features

- ✅ Real-time face detection and measurement using MediaPipe Face Mesh
- ✅ NIOSH 5-category classification (Small, Medium, Large, Long/Narrow, Short/Wide)
- ✅ Based on 2003 NIOSH anthropometric survey of 3,997 subjects
- ✅ Integration with NIOSH digital headform reference data
- ✅ Step-by-step OSHA user seal check protocol
- ✅ Downloadable fitting report
- ✅ Interactive web interface using Streamlit

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam
- Windows, macOS, or Linux

### Setup

1. **Clone or download this repository**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Place data files in the same directory:**
   - `RespiratorUsersData-all-subjects.csv`
   - `RespiratorUsers-scanned-subjects.csv`
   - STL headform files (optional, for enhanced reference)

## Usage

### Running the Application

```bash
streamlit run mask_fitting_app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Workflow

#### Step 1: Face Scan
- Position yourself 18-24 inches from webcam
- Ensure good lighting (face a window or light source)
- Remove glasses and pull hair back
- Maintain neutral expression
- Click to capture image
- System automatically detects face and extracts measurements

#### Step 2: Analysis & Recommendation
- View your facial measurements
- See your NIOSH face size category
- Review recommended mask brands, models, and sizes
- Compare your measurements to reference headform ranges

#### Step 3: Fit Test Protocol
- Complete pre-donning checklist
- Perform positive pressure test
- Perform negative pressure test
- Download fitting report

## NIOSH Face Size Categories

The system uses 5 categories based on NIOSH anthropometric data:

| Category | Bizygomatic Breadth | Menton-Sellion Length | Population Percentile |
|----------|---------------------|------------------------|----------------------|
| Small | 125-135 mm | 105-115 mm | ~10-25th |
| Medium | 135-145 mm | 115-125 mm | ~25-75th |
| Large | 145-160 mm | 125-135 mm | ~75-95th |
| Long/Narrow | 125-140 mm | 125-140 mm | Specialized |
| Short/Wide | 145-165 mm | 105-120 mm | Specialized |

## Technical Details

### Facial Measurements

The system extracts four key measurements:
1. **Bizygomatic Breadth** - Maximum width of face at cheekbones
2. **Menton-Sellion Length** - Vertical distance from chin to nose bridge
3. **Face Width** - Overall horizontal width
4. **Face Length** - Overall vertical height

### Measurement Method

- Uses MediaPipe Face Mesh (468 3D facial landmarks)
- Calculates distances between specific landmark points
- Calibration factor: assumes average face width ~140mm
- **Note:** For production use, calibration should be improved using a reference object

### Accuracy Considerations

- Webcam-based measurements are approximate
- Lighting and camera quality affect accuracy
- Best used as preliminary screening tool
- Professional fit testing may still be required for workplace compliance

## Data Sources

### NIOSH Anthropometric Survey (2003)
- 3,997 subjects measured with traditional anthropometry
- 1,013 subjects scanned with 3D head scanner
- Stratified sampling across age, gender, and race/ethnicity
- Data weighted to represent U.S. worker population

### ISO Digital Headforms
- 5 reference headforms representing each size category
- Each headform averages 5 representative individuals
- Incorporated into ISO 16976-2 standard
- Available in STL and IGS formats

## Example Mask Database

The application includes an example database of N95 respirators. In production, this should be expanded to include:
- More mask brands and models
- Different respirator types (N95, P100, half-face, full-face)
- Manufacturer-specific size charts
- Updated NIOSH approval numbers
- Supplier information

## Compliance Notes

### OSHA 1910.134 Requirements
This system provides **preliminary fitting guidance only**. OSHA regulations require:
- Initial fit testing by qualified person
- Annual re-testing
- Re-testing when physical changes occur
- Quantitative or qualitative fit testing (not just user seal check)

### Appropriate Use
✅ Pre-screening to identify likely mask sizes  
✅ Educational tool for understanding face sizing  
✅ Preliminary selection before professional fit test  
❌ Replacement for OSHA-required fit testing  
❌ Certification for use in hazardous environments  

## Future Enhancements

Potential improvements for production version:

1. **Calibration System**
   - Use reference card or object for accurate scaling
   - Support for depth cameras (Intel RealSense, etc.)
   - Multiple image capture for measurement averaging

2. **3D Headform Visualization**
   - Display appropriate STL headform for each category
   - Overlay user measurements on 3D model
   - Interactive 3D comparison

3. **Enhanced Database**
   - Comprehensive mask catalog with images
   - Integration with manufacturer APIs
   - Real-time inventory checking
   - Purchase links

4. **Fit Test Documentation**
   - Photo capture during fit testing
   - Digital signature capability
   - OSHA-compliant report generation
   - Record retention system

5. **Advanced Analytics**
   - Population statistics
   - Fit success rates by category
   - Mask performance tracking
   - Demographic analysis

## File Structure

```
mask_fitting_app.py          # Main Streamlit application
requirements.txt             # Python dependencies
README.md                    # This file
stl_parser.py               # STL headform analysis tool
RespiratorUsersData-all-subjects.csv   # NIOSH anthropometric data
RespiratorUsers-scanned-subjects.csv   # 3D scan subset data
large_symmetry.stl          # NIOSH large headform (example)
```

## Troubleshooting

### Camera Not Working
- Check camera permissions in system settings
- Close other applications using webcam
- Try restarting the browser

### Face Not Detected
- Improve lighting conditions
- Move closer to or farther from camera
- Ensure face is fully visible and unobstructed
- Remove glasses and face coverings

### Measurements Seem Incorrect
- Verify lighting conditions
- Ensure proper distance from camera (18-24 inches)
- Take multiple captures and compare results
- Consider using calibration object (credit card, etc.)

## References

1. NIOSH (2020). "NIOSH Anthropometric Data and ISO Digital Headforms." DOI: 10.26616/NIOSHRD-1013-2020-0

2. Zhuang, Z., et al. (2010). "Development of more effective respirator fit test panels." Applied Ergonomics, 41(6), 899-908.

3. OSHA 29 CFR 1910.134 - Respiratory Protection Standard

4. ISO 16976-2:2015 - Respiratory protective devices - Human factors - Part 2: Anthropometrics

5. NIOSH (2018). "NIOSH Guide to the Selection and Use of Particulate Respirators." DHHS (NIOSH) Publication No. 96-101.

## License

This is a prototype demonstration tool. NIOSH data is public domain. Check specific licensing for production use.

## Contact & Support

For questions about:
- **NIOSH data**: Contact NIOSH NPPTL at (412) 386-6111
- **Respirator standards**: Consult OSHA 1910.134
- **This application**: Educational/demonstration purposes only

## Disclaimer

This application is provided for educational and preliminary screening purposes only. It does NOT replace professional fit testing required by OSHA regulations. Always consult with qualified safety professionals and follow all applicable workplace safety regulations.

Respirator selection and use should always follow manufacturer guidelines, NIOSH recommendations, and OSHA requirements. Having effective respiratory protection can be a matter of life and death.
