// Face Verification Module
const FaceVerification = {
    stream: null,
    video: null,
    canvas: null,
    statusElement: null,
    verifyButton: null,
    csrfToken: null,
    maxAttempts: 3,
    currentAttempts: 0,
    autoVerify: false,
    onSuccess: null,
    onMaxAttempts: null,

    async initialize(options) {
        console.log('Initializing FaceVerification with options:', options);
        
        try {
            // Store options
            this.video = document.getElementById(options.videoId);
            this.canvas = document.getElementById(options.canvasId);
            this.statusElement = document.getElementById(options.statusElementId);
            this.verifyButton = document.getElementById(options.verifyButtonId);
            this.csrfToken = options.csrfToken;
            this.maxAttempts = options.maxAttempts || 3;
            this.autoVerify = options.autoVerify || false;
            this.onSuccess = options.onSuccess;
            this.onMaxAttempts = options.onMaxAttempts;

            console.log('Checking required elements:', {
                video: this.video,
                canvas: this.canvas,
                statusElement: this.statusElement,
                verifyButton: this.verifyButton
            });

            if (!this.video) {
                throw new Error('Video element not found');
            }

            if (!this.canvas) {
                throw new Error('Canvas element not found');
            }

            // Start camera
            await this.startCamera();

            // Enable verify button
            if (this.verifyButton) {
                this.verifyButton.disabled = false;
                this.verifyButton.addEventListener('click', () => this.verifyFace());
            }

            // Auto verify if enabled
            if (this.autoVerify) {
                this.verifyFace();
            }
        } catch (error) {
            console.error('Error in FaceVerification.initialize:', error);
            this.showStatus(error.message, 'danger');
            throw error;
        }
    },

    async startCamera() {
        console.log('Starting camera...');
        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Camera API not supported in this browser');
            }

            console.log('Requesting camera access...');
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                } 
            });

            console.log('Camera access granted, setting up video element...');
            this.video.srcObject = this.stream;
            
            // Wait for video to be ready
            await new Promise((resolve, reject) => {
                this.video.onloadedmetadata = () => {
                    console.log('Video metadata loaded, starting playback...');
                    this.video.play()
                        .then(() => {
                            console.log('Video playback started');
                            resolve();
                        })
                        .catch(error => {
                            console.error('Error starting video playback:', error);
                            reject(error);
                        });
                };
                
                // Add error handler
                this.video.onerror = (error) => {
                    console.error('Video element error:', error);
                    reject(error);
                };
            });

            this.showStatus('Camera initialized successfully', 'success');
            console.log('Camera initialization complete');
        } catch (err) {
            console.error('Error starting camera:', err);
            this.showStatus(this.getErrorMessage(err), 'danger');
            throw err;
        }
    },

    stopCamera() {
        console.log('Stopping camera...');
        if (this.stream) {
            this.stream.getTracks().forEach(track => {
                console.log('Stopping track:', track.kind);
                track.stop();
            });
            this.stream = null;
            this.video.srcObject = null;
            console.log('Camera stopped');
        }
    },

    async verifyFace() {
        if (this.currentAttempts >= this.maxAttempts) {
            this.showStatus('Maximum verification attempts reached', 'danger');
            if (this.onMaxAttempts) {
                this.onMaxAttempts();
            }
            return;
        }

        this.currentAttempts++;
        this.showStatus('Verifying face...', 'info');

        try {
            // Capture frame from video
            const context = this.canvas.getContext('2d');
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            context.drawImage(this.video, 0, 0);

            // Convert to base64
            const imageData = this.canvas.toDataURL('image/jpeg');

            // Send to server for verification
            const response = await fetch('/api/verify-face/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.csrfToken
                },
                body: JSON.stringify({ face_data: imageData })
            });

            const result = await response.json();

            if (result.success) {
                this.showStatus('Face verification successful!', 'success');
                if (this.onSuccess) {
                    this.onSuccess(result);
                }
            } else {
                this.showStatus(result.message || 'Face verification failed', 'danger');
            }
        } catch (err) {
            console.error('Error during face verification:', err);
            this.showStatus('Error during face verification', 'danger');
        }
    },

    showStatus(message, type) {
        console.log(`Status update [${type}]:`, message);
        if (this.statusElement) {
            this.statusElement.textContent = message;
            this.statusElement.className = `alert alert-${type}`;
            this.statusElement.classList.remove('d-none');
        }
    },

    getErrorMessage(err) {
        console.error('Camera error:', err);
        if (err.name === 'NotAllowedError') {
            return 'Camera access denied. Please allow camera access in your browser settings.';
        } else if (err.name === 'NotFoundError') {
            return 'No camera found. Please connect a camera and try again.';
        } else if (err.name === 'NotReadableError') {
            return 'Camera is already in use by another application.';
        } else if (err.name === 'OverconstrainedError') {
            return 'Camera does not support the requested resolution.';
        } else {
            return 'Camera access denied or not available.';
        }
    }
};