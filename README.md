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