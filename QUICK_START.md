# Quick Start Guide

## Getting Started in 3 Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Application

**On Windows:**
- Double-click `start_app.bat`
- OR open Command Prompt and run: `streamlit run mask_fitting_app.py`

**On Mac/Linux:**
- Run `./start_app.sh` in terminal
- OR run: `streamlit run mask_fitting_app.py`

### 3. Use the Application
- Application opens in browser at `http://localhost:8501`
- Follow the 3-step workflow:
  1. **Face Scan** - Capture your face with webcam
  2. **Analysis** - View measurements and mask recommendations
  3. **Fit Test** - Follow OSHA seal check protocol

## System Requirements

- **Python:** 3.8 or higher
- **Webcam:** Built-in or USB webcam
- **RAM:** 4GB minimum (8GB recommended)
- **OS:** Windows 10/11, macOS 10.15+, or Linux

## Troubleshooting

### Camera Issues
- **Permission denied:** Check system camera permissions
- **Camera in use:** Close other apps using webcam (Zoom, Skype, etc.)
- **No camera found:** Check webcam connection

### Installation Issues
- **pip not found:** Install pip with `python -m ensurepip`
- **Permission error:** Try `pip install --user -r requirements.txt`
- **Module not found:** Reinstall dependencies

### Face Detection Issues
- **No face detected:**
  - Improve lighting (face toward window/light)
  - Move 18-24 inches from camera
  - Remove glasses and pull hair back
  - Ensure face is centered and fully visible

### Performance Issues
- Close unnecessary browser tabs
- Use Chrome or Firefox for best performance
- Reduce camera resolution in system settings if needed

## File Organization

```
mask_fitting_system/
├── mask_fitting_app.py          # Main application
├── requirements.txt              # Dependencies
├── README.md                     # Full documentation
├── QUICK_START.md               # This file
├── start_app.sh                 # Linux/Mac startup
├── start_app.bat                # Windows startup
├── stl_utilities.py             # STL processing tools
├── RespiratorUsersData-all-subjects.csv
├── RespiratorUsers-scanned-subjects.csv
└── large_symmetry.stl           # Example headform
```

## Data Files

The application requires these CSV files (included):
- `RespiratorUsersData-all-subjects.csv` - Full NIOSH dataset
- `RespiratorUsers-scanned-subjects.csv` - 3D scan subset

Optional STL files for reference:
- `small_symmetry.stl`
- `medium_symmetry.stl`
- `large_symmetry.stl`
- `long_narrow_symmetry.stl`
- `short_wide_symmetry.stl`

## Tips for Best Results

### Camera Setup
- **Distance:** 18-24 inches from camera
- **Lighting:** Bright, even lighting on face (face toward light source)
- **Background:** Plain, uncluttered background
- **Position:** Face camera straight on, not at an angle

### Before Scanning
- Remove glasses
- Pull hair back from face
- Remove any face coverings
- Use neutral expression (no smiling)
- Ensure face is clean-shaven (for respirator use)

### Understanding Results

**Measurements:**
- Bizygomatic Breadth = Face width at cheekbones
- Menton-Sellion = Chin to nose bridge distance

**Categories:**
- Small: 10-25th percentile
- Medium: 25-75th percentile (most common)
- Large: 75-95th percentile
- Long/Narrow: Specialized fit
- Short/Wide: Specialized fit

**Recommendations:**
- Top 3 mask models for your category
- Expected fit scores
- Specific sizes to try

## Important Safety Notes

⚠️ **This is a screening tool only**

- Does NOT replace OSHA-required fit testing
- Professional fit testing may be required
- Always follow workplace safety protocols
- Consult safety professionals for workplace use

✅ **Appropriate for:**
- Pre-screening before buying masks
- Educational purposes
- Understanding face sizing
- Preliminary selection

❌ **NOT appropriate for:**
- OSHA compliance certification
- Use in hazardous environments without additional testing
- Replacing professional fit testing

## Next Steps After Fitting

1. **Obtain Recommended Mask**
   - Purchase one of the recommended models/sizes
   - Try multiple brands if possible

2. **Perform User Seal Check**
   - Follow the guided protocol in Step 3
   - Do this EVERY time you wear the mask

3. **Professional Fit Testing**
   - If required by employer/OSHA
   - Contact occupational health professional
   - Quantitative or qualitative testing

4. **Keep Records**
   - Download fitting report
   - Document successful fit tests
   - Note which masks work best

## Support

### Documentation
- Full README.md for complete documentation
- OSHA 1910.134 for regulatory requirements
- NIOSH website for respirator guidance

### Technical Support
This is a prototype/educational tool. For:
- NIOSH data questions: (412) 386-6111
- Respirator selection: Consult safety professional
- OSHA compliance: Contact your EHS department

## Updating the Mask Database

The application includes an example mask database. To add more masks:

1. Open `mask_fitting_app.py`
2. Find the `MASK_DATABASE` dictionary
3. Add entries in this format:
```python
'category_name': [
    {
        'brand': 'Brand Name',
        'model': 'Model Number',
        'size': 'Size',
        'fit_score': 95  # Expected fit percentage
    },
    # Add more masks...
]
```

## Advanced Features

### STL Headform Integration
If you have all 5 NIOSH STL headforms:

1. Place them in the application directory
2. Run: `python stl_utilities.py`
3. This creates `headform_references.json`
4. App will automatically use enhanced reference data

### Customization
Edit these sections in `mask_fitting_app.py`:
- `NIOSH_HEADFORMS` - Adjust category ranges
- `MASK_DATABASE` - Add/remove mask models
- CSS styles - Modify appearance
- Fit test protocol - Customize instructions

## License & Disclaimer

**License:** Educational/prototype use
**NIOSH Data:** Public domain
**Disclaimer:** For educational purposes only. Does not replace professional fit testing or OSHA compliance requirements.

---

**Version:** 1.0  
**Last Updated:** February 2026  
**Based on:** NIOSH 2003 Anthropometric Survey
