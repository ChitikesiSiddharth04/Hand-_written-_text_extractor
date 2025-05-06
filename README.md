# Form Processing OCR System

An advanced Optical Character Recognition (OCR) system for processing banking/financial application forms. This system automatically extracts and processes information from PDF forms using computer vision and deep learning techniques.

## Features

- PDF to image conversion
- Automated form field detection
- Checkbox state recognition
- Text extraction using TrOCR
- Multi-page form processing
- Structured data output
- Excel export functionality

## Prerequisites

- Python 3.7+
- OpenCV
- PyTorch
- Transformers
- PDF2Image
- Pandas
- PIL (Python Imaging Library)
- Poppler (for PDF processing)

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Install Poppler:
- Windows: Download and install from [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/)
- Linux: `sudo apt-get install poppler-utils`
- Mac: `brew install poppler`

## Project Structure

```
├── all_pages_code.py          # Main orchestration file
├── final1.py to final5.py     # Core processing functions
├── sixth_page.py to eleventh_page.py  # Page-specific processing
├── util.py                    # Utility functions
└── requirements.txt           # Project dependencies
```

## Usage

1. Place your PDF form in the input directory
2. Run the main script:
```bash
python all_pages_code.py
```

3. The processed data will be exported to Excel files in the output directory

## Key Components

### Image Processing
- Line removal
- Noise reduction
- Contour detection
- Field isolation

### OCR Capabilities
- Text recognition using TrOCR
- Checkbox detection
- Field label extraction
- Confidence scoring

### Form Processing
- Multiple page handling
- Different field types
- Structured data extraction
- Excel output generation

## Supported Form Fields

- Personal Information
  - Name
  - Gender
  - Race
  - Marital status
  - Education level
  - Residency status

- Financial Information
  - Credit card details
  - Loan information
  - Employment details
  - Tax information

- Additional Fields
  - Identification numbers
  - Contact information
  - Declarations
  - Signatures

## Output Format

The system generates Excel files with the following structure:
- Multiple sheets (one per page)
- Field names and extracted values
- Organized data format

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments

- TrOCR model for text recognition
- OpenCV for image processing
- PDF2Image for PDF conversion
- Pandas for data handling

