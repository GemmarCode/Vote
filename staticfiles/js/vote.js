// Global variables
let csrftoken;
let currentPositionIndex = 0;
let positions = [];
let selectedCandidate = null;
let userVotes = [];
let currentPosition = null;

document.addEventListener('DOMContentLoaded', function() {
    // Initialize CSRF token
    csrftoken = document.querySelector('[name=csrfmiddlewaretoken]').value;

    // Function to initialize face verification
    async function initFaceVerification() {
        try {
            console.log('Initializing face verification...');
            
            // Wait for the modal to be fully shown
            const modal = document.getElementById('faceVerificationModal');
            await new Promise(resolve => {
                modal.addEventListener('shown.bs.modal', resolve, { once: true });
            });
            
            console.log('Modal shown, initializing FaceVerification...');
            
            // Initialize the FaceVerification module
            await FaceVerification.initialize({
                videoId: 'video',
                canvasId: 'canvas',
                statusElementId: 'verification-status',
                verifyButtonId: 'verifyButton',
                csrfToken: csrftoken,
                maxAttempts: 3,
                autoVerify: true,
                onSuccess: handleSuccessfulVerification
            });
            
            console.log('Face verification initialized successfully');
        } catch (err) {
            console.error('Error initializing face verification:', err);
            showVerificationStatus('Error initializing camera: ' + err.message, 'danger');
        }
    }

    console.log('DOM loaded, starting verification process');
    
    // Show face verification modal
    const faceVerificationModal = new bootstrap.Modal(document.getElementById('faceVerificationModal'), {
        backdrop: 'static',
        keyboard: false
    });
    faceVerificationModal.show();
    
    // Wait for modal to be fully shown before initializing face verification
    document.getElementById('faceVerificationModal').addEventListener('shown.bs.modal', function() {
        // Small delay to ensure elements are rendered
        setTimeout(initFaceVerification, 100);
    });
    
    // Code entry functionality
    document.getElementById('showCodeEntry').addEventListener('click', function(e) {
        e.preventDefault();
        document.getElementById('webcamSection').style.display = 'none';
        document.getElementById('codeEntryForm').style.display = 'block';
        
        // Stop the webcam when switching to code entry
        if (typeof FaceVerification !== 'undefined') {
            FaceVerification.stopCamera();
        }
    });
    
    // Back to webcam functionality
    document.getElementById('backToWebcam').addEventListener('click', async function() {
        document.getElementById('codeEntryForm').style.display = 'none';
        document.getElementById('webcamSection').style.display = 'block';
        
        // Restart the webcam
        if (typeof FaceVerification !== 'undefined') {
            try {
                await FaceVerification.startCamera();
            } catch (err) {
                console.error('Error restarting camera:', err);
                showVerificationStatus('Error starting camera. Please refresh the page.', 'danger');
            }
        }
    });
});

// Helper function to show verification status
function showVerificationStatus(message, type) {
    const statusElement = document.getElementById('verification-status');
    if (statusElement) {
        statusElement.textContent = message;
        statusElement.className = `alert alert-${type}`;
        statusElement.classList.remove('d-none');
    }
}

// Handle successful verification
function handleSuccessfulVerification(result) {
    console.log('Verification successful:', result);
    
    // Store user profile data
    const userProfile = {
        id: result.user_profile_id,
        studentNumber: result.student_number,
        college: result.college,
        department: result.department,
        yearLevel: result.year_level
    };
    
    // Store profile in session storage
    sessionStorage.setItem('verifiedUserProfile', JSON.stringify(userProfile));
    
    // Hide verification modal
    const verificationModal = bootstrap.Modal.getInstance(document.getElementById('faceVerificationModal'));
    if (verificationModal) {
        verificationModal.hide();
    }
    
    // Check if voting is available
    fetch('/api/check-voting-status/', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrftoken
        },
        credentials: 'same-origin'
    })
    .then(response => response.json())
    .then(result => {
        if (result.is_voting_open) {
            startVotingProcess();
        } else {
            showVotingNotAvailableModal(result.message || 'Voting is currently closed.');
        }
    })
    .catch(error => {
        console.error('Error checking voting status:', error);
        showVotingNotAvailableModal('Error checking voting status. Please try again later.');
    });
}

// Show voting not available modal
function showVotingNotAvailableModal(message) {
    const modal = new bootstrap.Modal(document.getElementById('votingNotAvailableModal'));
    const messageElement = document.getElementById('voting-status-message');
    if (messageElement) {
        messageElement.textContent = message;
    }
    modal.show();
}

// Start the voting process
function startVotingProcess() {
    console.log('Starting voting process...');
    
    // Get user profile from session storage
    const userProfileJson = sessionStorage.getItem('verifiedUserProfile');
    if (!userProfileJson) {
        console.error('No verified user profile found in session storage');
        alert('Error: User authentication required. Please try again.');
        window.location.href = '/';
        return;
    }
    
    const userProfile = JSON.parse(userProfileJson);
    console.log('User profile:', userProfile);
    
    // Initialize empty votes array
    userVotes = [];
    localStorage.setItem('userVotes', JSON.stringify(userVotes));
    
    // Show voting modal
    const votingModal = new bootstrap.Modal(document.getElementById('votingModal'), {
        backdrop: 'static',
        keyboard: false
    });
    votingModal.show();
    
    // Load positions and candidates
    loadPositionsAndCandidates(userProfile);
}

// Load positions and candidates
function loadPositionsAndCandidates(userProfile) {
    console.log('Loading positions and candidates...');
    
    // Fetch positions and candidates from the server
    fetch('/api/get-positions-candidates/', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrftoken
        },
        credentials: 'same-origin'
    })
    .then(response => response.json())
    .then(data => {
        console.log('Received positions and candidates:', data);
        
        // Store positions
        positions = data.positions || [];
        
        // Update UI
        document.getElementById('totalPositions').textContent = positions.length;
        
        if (positions.length === 0) {
            console.error('No positions/candidates loaded');
            alert('Error: No candidates found for voting.');
            return;
        }
        
        // Start with first position
        currentPositionIndex = 0;
        showCurrentPosition();
    })
    .catch(error => {
        console.error('Error loading positions and candidates:', error);
        alert('Error loading voting data. Please try again later.');
    });
}

// Show current position and its candidates
function showCurrentPosition() {
    if (currentPositionIndex >= positions.length) {
        console.log('All positions voted');
        showCompletionModal();
        return;
    }
    
    currentPosition = positions[currentPositionIndex];
    console.log('Showing position:', currentPosition);
    
    // Update progress bar
    const progress = ((currentPositionIndex + 1) / positions.length) * 100;
    document.querySelector('.progress-bar').style.width = `${progress}%`;
    document.querySelector('.progress-bar').setAttribute('aria-valuenow', progress);
    
    // Update position counter
    document.getElementById('currentPositionNumber').textContent = currentPositionIndex + 1;
    
    // Update position title
    document.getElementById('positionTitle').querySelector('h3').textContent = currentPosition.title;
    
    // Clear previous candidates
    const candidatesContainer = document.getElementById('candidatesContainer');
    candidatesContainer.innerHTML = '';
    
    // Add candidates
    currentPosition.candidates.forEach(candidate => {
        const candidateCard = createCandidateCard(candidate);
        candidatesContainer.appendChild(candidateCard);
    });
}

// Create candidate card
function createCandidateCard(candidate) {
    const col = document.createElement('div');
    col.className = 'col-md-6';
    
    const card = document.createElement('div');
    card.className = 'card h-100 candidate-card';
    card.style.cursor = 'pointer';
    card.onclick = () => selectCandidate(candidate);
    
    const cardBody = document.createElement('div');
    cardBody.className = 'card-body text-center';
    
    const name = document.createElement('h5');
    name.className = 'card-title mb-2';
    name.textContent = candidate.name;
    
    const party = document.createElement('p');
    party.className = 'card-text text-muted mb-2';
    party.textContent = candidate.party || 'Independent';
    
    cardBody.appendChild(name);
    cardBody.appendChild(party);
    card.appendChild(cardBody);
    col.appendChild(card);
    
    return col;
}

// Select a candidate
function selectCandidate(candidate) {
    console.log('Selected candidate:', candidate);
    selectedCandidate = candidate;
    
    // Show confirmation modal
    const modal = new bootstrap.Modal(document.getElementById('voteConfirmationModal'));
    const details = document.getElementById('confirmationCandidateDetails');
    details.innerHTML = `
        <h4 class="mb-2">${candidate.name}</h4>
        <p class="text-muted mb-0">${candidate.party || 'Independent'}</p>
        <p class="mb-0">for ${currentPosition.title}</p>
    `;
    
    // Add button handlers
    document.getElementById('confirmVoteButton').onclick = confirmVote;
    document.getElementById('cancelVoteButton').onclick = cancelVote;
    
    modal.show();
}

// Record vote
function recordVote(candidate) {
    console.log('Recording vote for:', candidate);
    
    // Add vote to array
    userVotes.push({
        position: currentPosition.title,
        candidate: candidate.id
    });
    
    // Save to local storage
    localStorage.setItem('userVotes', JSON.stringify(userVotes));
    
    // Move to next position
    currentPositionIndex++;
    showCurrentPosition();
}

// Show completion modal
function showCompletionModal() {
    console.log('Showing completion modal');
    
    // Hide voting modal
    const votingModal = bootstrap.Modal.getInstance(document.getElementById('votingModal'));
    if (votingModal) {
        votingModal.hide();
    }
    
    // Submit votes to server
    fetch('/api/submit-votes/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrftoken
        },
        body: JSON.stringify({
            votes: userVotes
        }),
        credentials: 'same-origin'
    })
    .then(response => response.json())
    .then(result => {
        if (result.success) {
            // Show completion modal
            const completionModal = new bootstrap.Modal(document.getElementById('completionModal'));
            completionModal.show();
            
            // Clear stored votes
            localStorage.removeItem('userVotes');
        } else {
            throw new Error(result.message || 'Failed to submit votes');
        }
    })
    .catch(error => {
        console.error('Error submitting votes:', error);
        alert('Error submitting votes. Please try again later.');
    });
}

// Confirm vote
function confirmVote() {
    console.log('Confirming vote for:', selectedCandidate);
    
    // Hide confirmation modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('voteConfirmationModal'));
    if (modal) {
        modal.hide();
    }
    
    // Record the vote
    recordVote(selectedCandidate);
}

// Cancel vote
function cancelVote() {
    console.log('Cancelling vote');
    
    // Hide confirmation modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('voteConfirmationModal'));
    if (modal) {
        modal.hide();
    }
    
    // Clear selected candidate
    selectedCandidate = null;
} 