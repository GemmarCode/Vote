# Smart Vote System

A web-based voting system with face recognition capabilities for university elections.

## Features

- Face recognition login
- Traditional username/password login
- Admin panel for managing users and elections
- Real-time voting system
- Candidate management
- Election results tracking
- Secure voting process

## Prerequisites

- Python 3.8 or higher
- MySQL Server
- Webcam (for face recognition)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SmartVoteSystem.git
cd SmartVoteSystem
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up MySQL database:
```sql
CREATE DATABASE voting7;
```

5. Configure the database settings in `smartvote/settings.py`:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'voting7',
        'USER': 'root',
        'PASSWORD': '',  # Set your MySQL password here
        'HOST': 'localhost',
        'PORT': '3306',
    }
}
```

6. Apply database migrations:
```bash
python manage.py migrate
```

7. Create a superuser:
```bash
python manage.py createsuperuser
```

## Running the Application

1. Start the development server:
```bash
python manage.py runserver
```

2. Access the application:
- Main application: http://127.0.0.1:8000/
- Admin panel: http://127.0.0.1:8000/admin/
- Login page: http://127.0.0.1:8000/login/

## Project Structure

- `user/` - Main application for user management and voting
- `admin_panel/` - Admin interface for managing the system
- `smartvote/` - Project configuration
- `media/` - User-uploaded files
- `static/` - Static files (CSS, JavaScript, images)

## Security Features

- Face recognition for secure login
- SSL/HTTPS support
- CSRF protection
- Secure session handling
- Password validation
- XSS protection

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Enhanced Face Verification System

## Overview
This system features an improved face verification system for secure voter authentication. The enhancements include:

1. **Multiple Verification Attempts**: The system automatically tries up to 3 verification attempts for better accuracy.
2. **Advanced Face Detection**: Uses OpenCV's Haar Cascade to detect and focus on face areas.
3. **Improved Lighting Normalization**: Implements CLAHE (Contrast Limited Adaptive Histogram Equalization) for better handling of varying lighting conditions.
4. **Multiple Matching Methods**: Uses a combination of techniques, including structural similarity, histogram comparison, and face recognition libraries.
5. **Comprehensive Error Handling**: Detailed logging and error reporting for debugging.
6. **Debug Image Saving**: Automatically saves captured and processed images for troubleshooting.
7. **Modular JavaScript Interface**: A reusable face verification module for consistent implementation.

## Face Verification Process

### Server-Side
The backend implements a robust verification process:
1. Captures and preprocesses face images
2. Attempts multiple verification methods
3. Selects the best match across all attempts
4. Provides detailed feedback on verification results

### Client-Side
The frontend JavaScript module (`face-verification.js`) provides:
1. Camera initialization and management
2. Multiple verification attempts
3. User feedback on verification status
4. Visual effects during verification

## Usage Instructions

### Adding the JavaScript Module
Include the JavaScript file in your HTML:
```html
<script src="{% static 'js/face-verification.js' %}"></script>
```

### Initializing Face Verification
```javascript
// Initialize with options
await FaceVerification.initialize({
    videoId: 'video',                     // Video element ID
    canvasId: 'canvas',                   // Canvas element ID
    statusElementId: 'verification-status', // Status display element ID
    verifyButtonId: 'verifyButton',       // Verification button ID
    csrfToken: csrfToken,                 // CSRF token for requests
    maxAttempts: 3,                       // Maximum verification attempts
    autoVerify: false,                    // Auto-start verification
    onSuccess: function(data) {
        // Handle successful verification
    },
    onMaxAttempts: function(data) {
        // Handle maximum attempts reached
    }
});
```

### Stopping the Camera
```javascript
FaceVerification.stopCamera();
```

### Common Issues & Troubleshooting
1. **Verification Fails**: Ensure good lighting and face positioning
2. **Camera Not Starting**: Check browser permissions
3. **Slow Performance**: Reduce video resolution in the options

## API Endpoints

### Verify Face
- **URL**: `/api/verify-face/`
- **Method**: POST
- **Body**: JSON with face_data (base64 image)
- **Response**: JSON with verification results

## Testing
For best results:
1. Ensure good lighting conditions
2. Position face clearly in the frame
3. Remove glasses or face coverings
4. Maintain a neutral expression

## Debug Mode
Debug images are saved in the `media/debug` directory and include:
- Original captured images
- Preprocessed faces
- Verification log files with detailed scoring 