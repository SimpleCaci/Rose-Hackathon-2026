import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

let poseLandmarker;
let webcamRunning = false;
let calibrating = false;
let videoElement;
let canvasElement;
let canvasCtx;
let selectedPose = 'tree';
let userHeight = null;
let isCalibrated = false;
let currentStream = null;

// DOM Elements
const startButton = document.getElementById('startButton');
const poseSelect = document.getElementById('poseSelect');
const feedbackText = document.getElementById('feedbackText');
const accuracyDisplay = document.getElementById('accuracy');
const calibrateButton = document.getElementById('calibrateButton');
const calibrationStatus = document.getElementById('calibrationStatus');

// Reference poses with wider ranges (normalized to body proportions)
const referencePoses = {
    tree: {
        name: "Tree Pose",
        description: "Stand on one leg, place other foot on inner thigh, hands above head",
        landmarks: {
            // Standing leg (right) should be straight
            rightKnee: { angle: 180, tolerance: 15, points: [24, 26, 28] }, // hip-knee-ankle
            // Bent leg (left) knee should be out to side
            leftKnee: { angle: 90, tolerance: 20, points: [23, 25, 27] },
            // Arms raised above head
            leftElbow: { angle: 180, tolerance: 20, points: [11, 13, 15] }, // shoulder-elbow-wrist
            rightElbow: { angle: 180, tolerance: 20, points: [12, 14, 16] },
            // Arms should be vertical (shoulder-elbow vs vertical)
            leftArmVertical: { angle: 10, tolerance: 20, points: [11, 13, 'vertical'] },
            rightArmVertical: { angle: 10, tolerance: 20, points: [12, 14, 'vertical'] }
        }
    },
    warrior: {
        name: "Warrior II Pose",
        description: "Lunge position, arms extended to sides",
        landmarks: {
            // Front knee bent
            leftKnee: { angle: 90, tolerance: 20, points: [23, 25, 27] },
            // Back leg straight
            rightKnee: { angle: 180, tolerance: 15, points: [24, 26, 28] },
            // Arms extended horizontally
            leftElbow: { angle: 180, tolerance: 20, points: [11, 13, 15] },
            rightElbow: { angle: 180, tolerance: 20, points: [12, 14, 16] },
            // Arms should be horizontal
            leftArmHorizontal: { angle: 90, tolerance: 20, points: [11, 13, 'horizontal'] },
            rightArmHorizontal: { angle: 90, tolerance: 20, points: [12, 14, 'horizontal'] }
        }
    },
    downdog: {
        name: "Downward Dog",
        description: "Inverted V-shape, hands and feet on ground",
        landmarks: {
            // Both legs should be straight
            leftKnee: { angle: 180, tolerance: 20, points: [23, 25, 27] },
            rightKnee: { angle: 180, tolerance: 20, points: [24, 26, 28] },
            // Both arms should be straight
            leftElbow: { angle: 180, tolerance: 20, points: [11, 13, 15] },
            rightElbow: { angle: 180, tolerance: 20, points: [12, 14, 16] },
            // Hip angle (torso to legs)
            leftHip: { angle: 70, tolerance: 20, points: [11, 23, 25] }, // shoulder-hip-knee
            rightHip: { angle: 70, tolerance: 20, points: [12, 24, 26] }
        }
    }
};

// Joint names for feedback
const jointNames = {
    leftKnee: "left knee",
    rightKnee: "right knee",
    leftElbow: "left elbow",
    rightElbow: "right elbow",
    leftHip: "left hip",
    rightHip: "right hip",
    leftArmVertical: "left arm",
    rightArmVertical: "right arm",
    leftArmHorizontal: "left arm",
    rightArmHorizontal: "right arm"
};

// Body part priority (lower body first)
const feedbackPriority = [
    'rightKnee', 'leftKnee',
    'rightHip', 'leftHip',
    'rightElbow', 'leftElbow',
    'rightArmVertical', 'leftArmVertical',
    'rightArmHorizontal', 'leftArmHorizontal'
];

// Initialize MediaPipe
async function initializePoseLandmarker() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 1
    });
    
    console.log("Pose Landmarker initialized");
}

// Calculate angle between three points
function calculateAngle(a, b, c) {
    const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
    let angle = Math.abs(radians * 180.0 / Math.PI);
    
    if (angle > 180.0) {
        angle = 360 - angle;
    }
    
    return angle;
}

// Calculate angle from vertical or horizontal
function calculateSpecialAngle(a, b, type) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    
    if (type === 'vertical') {
        // Angle from vertical (0 degrees is straight up)
        const angle = Math.abs(Math.atan2(dx, -dy) * 180 / Math.PI);
        return angle;
    } else if (type === 'horizontal') {
        // Angle from horizontal (90 degrees is straight out to side)
        const angle = Math.abs(Math.atan2(dy, dx) * 180 / Math.PI);
        return Math.abs(90 - angle);
    }
    return 0;
}

// Calibrate user height
function calibrateHeight(landmarks) {
    if (!landmarks || landmarks.length === 0) return null;
    
    const pose = landmarks[0];
    
    // Make sure all required landmarks are visible
    const nose = pose[0];
    const leftAnkle = pose[27];
    const rightAnkle = pose[28];
    
    // Check visibility
    if (!nose || !leftAnkle || !rightAnkle) return null;
    
    // Calculate height from nose to average ankle position
    const avgAnkleY = (leftAnkle.y + rightAnkle.y) / 2;
    const height = Math.abs(avgAnkleY - nose.y);
    
    // Sanity check - height should be reasonable (not too small)
    if (height < 0.3) return null; // Minimum 30% of frame height
    
    return height;
}

// Check pose accuracy with color coding per joint
function checkPoseAccuracy(landmarks) {
    if (!landmarks || landmarks.length === 0) {
        return { accuracy: 0, feedback: [], jointStatus: {} };
    }

    const pose = landmarks[0];
    const reference = referencePoses[selectedPose];
    let totalScore = 0;
    let checkCount = 0;
    let feedback = [];
    let jointStatus = {}; // Track which joints are correct

    // Check each landmark in the reference pose
    for (const [jointName, jointData] of Object.entries(reference.landmarks)) {
        let actualAngle;
        
        if (jointData.points[2] === 'vertical' || jointData.points[2] === 'horizontal') {
            actualAngle = calculateSpecialAngle(
                pose[jointData.points[0]],
                pose[jointData.points[1]],
                jointData.points[2]
            );
        } else {
            actualAngle = calculateAngle(
                pose[jointData.points[0]],
                pose[jointData.points[1]],
                pose[jointData.points[2]]
            );
        }
        
        const targetAngle = jointData.angle;
        const tolerance = jointData.tolerance;
        const difference = Math.abs(actualAngle - targetAngle);
        
        // Determine if joint is in range
        const inRange = difference <= tolerance;
        jointStatus[jointName] = {
            inRange: inRange,
            difference: difference,
            points: jointData.points
        };
        
        // Calculate score
        if (inRange) {
            totalScore += 1.0;
        } else {
            // Partial credit for being close
            const partialScore = Math.max(0, 1.0 - (difference - tolerance) / tolerance);
            totalScore += partialScore;
        }
        checkCount++;
    }

    // Generate feedback in priority order (lower body first)
    for (const jointName of feedbackPriority) {
        if (jointStatus[jointName] && !jointStatus[jointName].inRange) {
            feedback.push(`Adjust your ${jointNames[jointName]}`);
            // Only show top 3 corrections
            if (feedback.length >= 3) break;
        }
    }

    const accuracy = checkCount > 0 ? (totalScore / checkCount) * 100 : 0;

    return { accuracy, feedback, jointStatus };
}

// Draw skeleton with color coding
function drawColorCodedSkeleton(landmarks, jointStatus) {
    const pose = landmarks[0];
    const reference = referencePoses[selectedPose];
    
    // Draw connections with color coding
    for (const [jointName, jointData] of Object.entries(reference.landmarks)) {
        const points = jointData.points;
        const inRange = jointStatus[jointName].inRange;
        
        // Skip special angle types for drawing
        if (points[2] === 'vertical' || points[2] === 'horizontal') continue;
        
        const color = inRange ? '#00ff00' : '#ff0000';
        const lineWidth = 8;
        
        // Draw line between points
        const p1 = pose[points[0]];
        const p2 = pose[points[1]];
        const p3 = pose[points[2]];
        
        canvasCtx.strokeStyle = color;
        canvasCtx.lineWidth = lineWidth;
        canvasCtx.lineCap = 'round';
        
        // Draw p1 to p2
        canvasCtx.beginPath();
        canvasCtx.moveTo(p1.x * canvasElement.width, p1.y * canvasElement.height);
        canvasCtx.lineTo(p2.x * canvasElement.width, p2.y * canvasElement.height);
        canvasCtx.stroke();
        
        // Draw p2 to p3
        canvasCtx.beginPath();
        canvasCtx.moveTo(p2.x * canvasElement.width, p2.y * canvasElement.height);
        canvasCtx.lineTo(p3.x * canvasElement.width, p3.y * canvasElement.height);
        canvasCtx.stroke();
    }
    
    // Draw circles at joints
    for (const [jointName, jointData] of Object.entries(reference.landmarks)) {
        const points = jointData.points;
        const inRange = jointStatus[jointName].inRange;
        const color = inRange ? '#00ff00' : '#ff0000';
        
        for (let i = 0; i < points.length; i++) {
            if (typeof points[i] === 'string') continue; // Skip 'vertical' and 'horizontal'
            
            const point = pose[points[i]];
            canvasCtx.fillStyle = color;
            canvasCtx.beginPath();
            canvasCtx.arc(
                point.x * canvasElement.width,
                point.y * canvasElement.height,
                10,
                0,
                2 * Math.PI
            );
            canvasCtx.fill();
        }
    }
}

// Stop current video stream
function stopVideoStream() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    if (videoElement.srcObject) {
        videoElement.srcObject = null;
    }
}

// Calibrate
async function calibrate() {
    if (!poseLandmarker) {
        alert("Please wait! Pose Landmarker is still loading");
        return;
    }

    calibrating = true;
    calibrationStatus.textContent = "Starting camera... Please wait";
    calibrationStatus.style.color = '#666';
    calibrateButton.disabled = true;

    const constraints = {
        video: {
            width: { ideal: 1280 },
            height: { ideal: 720 }
        }
    };

    try {
        // Stop any existing stream
        stopVideoStream();
        
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        currentStream = stream;
        videoElement.srcObject = stream;
        
        // Wait for video to be ready
        await new Promise((resolve) => {
            videoElement.onloadedmetadata = () => {
                videoElement.play();
                resolve();
            };
        });

        // Give user 3 seconds to get into T-pose
        calibrationStatus.textContent = "Get into T-pose position (arms out to sides)... 3";
        await new Promise(resolve => setTimeout(resolve, 1000));
        calibrationStatus.textContent = "Get into T-pose position (arms out to sides)... 2";
        await new Promise(resolve => setTimeout(resolve, 1000));
        calibrationStatus.textContent = "Get into T-pose position (arms out to sides)... 1";
        await new Promise(resolve => setTimeout(resolve, 1000));
        calibrationStatus.textContent = "Calibrating... Hold still!";
        
        // Start calibration
        performCalibration();
        
    } catch (error) {
        console.error("Error accessing webcam:", error);
        alert("Could not access webcam. Please check permissions.");
        calibrating = false;
        calibrateButton.disabled = false;
        calibrationStatus.textContent = "Calibration failed. Please try again.";
        calibrationStatus.style.color = '#dc3545';
        stopVideoStream();
    }
}

// Perform the actual calibration
function performCalibration() {
    let samplesCollected = 0;
    let heightSamples = [];
    const samplesToCollect = 10;
    let frameCount = 0;
    
    function calibrationLoop() {
        if (!calibrating) return;
        
        // Make sure video is ready
        if (videoElement.readyState >= 2) {
            canvasElement.width = videoElement.videoWidth;
            canvasElement.height = videoElement.videoHeight;
            
            const startTimeMs = performance.now();
            
            try {
                const results = poseLandmarker.detectForVideo(videoElement, startTimeMs);
                
                // Clear canvas
                canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
                
                if (results.landmarks && results.landmarks.length > 0) {
                    // Draw the pose so user can see
                    const drawingUtils = new DrawingUtils(canvasCtx);
                    for (const landmark of results.landmarks) {
                        drawingUtils.drawLandmarks(landmark, {
                            radius: 8,
                            color: '#00ff00',
                            fillColor: '#00ff00'
                        });
                        drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS, {
                            color: '#00ff00',
                            lineWidth: 4
                        });
                    }
                    
                    // Only collect samples every 5 frames to give variety
                    if (frameCount % 5 === 0) {
                        const height = calibrateHeight(results.landmarks);
                        if (height && height > 0) {
                            heightSamples.push(height);
                            samplesCollected++;
                            
                            // Update status to show progress
                            calibrationStatus.textContent = `Calibrating... ${samplesCollected}/${samplesToCollect} samples`;
                            
                            if (samplesCollected >= samplesToCollect) {
                                // Calculate average height
                                userHeight = heightSamples.reduce((a, b) => a + b, 0) / heightSamples.length;
                                isCalibrated = true;
                                calibrating = false;
                                
                                calibrationStatus.textContent = `✓ Calibrated successfully! Ready to practice.`;
                                calibrationStatus.style.color = '#28a745';
                                
                                calibrateButton.disabled = false;
                                startButton.disabled = false;
                                
                                // Stop video stream
                                setTimeout(() => {
                                    stopVideoStream();
                                    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
                                }, 1500);
                                
                                return; // Exit the loop
                            }
                        }
                    }
                    frameCount++;
                } else {
                    calibrationStatus.textContent = "No person detected. Please stand in frame with T-pose.";
                    calibrationStatus.style.color = '#ffc107';
                }
            } catch (error) {
                console.error("Calibration error:", error);
            }
        }
        
        // Continue loop
        if (calibrating) {
            requestAnimationFrame(calibrationLoop);
        }
    }
    
    // Start the calibration loop
    calibrationLoop();
    
    // Timeout after 20 seconds
    setTimeout(() => {
        if (calibrating && !isCalibrated) {
            calibrating = false;
            calibrationStatus.textContent = "Calibration timeout. Please try again.";
            calibrationStatus.style.color = '#dc3545';
            calibrateButton.disabled = false;
            stopVideoStream();
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        }
    }, 20000);
}

// Enable camera
async function enableCam() {
    if (!poseLandmarker) {
        alert("Please wait! Pose Landmarker is still loading");
        return;
    }

    if (!isCalibrated) {
        alert("Please calibrate first by standing in T-pose!");
        return;
    }

    if (webcamRunning) {
        webcamRunning = false;
        startButton.innerText = "Start Practice";
        stopVideoStream();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        return;
    }

    webcamRunning = true;
    startButton.innerText = "Stop Practice";

    const constraints = {
        video: {
            width: { ideal: 1280 },
            height: { ideal: 720 }
        }
    };

    try {
        stopVideoStream();
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        currentStream = stream;
        videoElement.srcObject = stream;
        videoElement.addEventListener("loadeddata", predictWebcam);
    } catch (error) {
        console.error("Error accessing webcam:", error);
        alert("Could not access webcam. Please check permissions.");
        webcamRunning = false;
        startButton.innerText = "Start Practice";
    }
}

// Predict webcam
async function predictWebcam() {
    if (!webcamRunning) return;

    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;

    let startTimeMs = performance.now();
    
    if (videoElement.currentTime > 0) {
        const results = poseLandmarker.detectForVideo(videoElement, startTimeMs);
        
        // Clear canvas
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        
        if (results.landmarks && results.landmarks.length > 0) {
            // Check pose accuracy
            const { accuracy, feedback, jointStatus } = checkPoseAccuracy(results.landmarks);
            
            // Draw color-coded skeleton
            drawColorCodedSkeleton(results.landmarks, jointStatus);
            
            // Update UI
            accuracyDisplay.textContent = `${Math.round(accuracy)}%`;
            
            if (feedback.length > 0) {
                feedbackText.innerHTML = feedback.map(f => `<div class="feedback-item">• ${f}</div>`).join('');
            } else {
                feedbackText.innerHTML = '<div class="feedback-item good">✓ Perfect form! Hold this pose!</div>';
            }

            // Color code accuracy
            accuracyDisplay.className = '';
            if (accuracy >= 85) {
                accuracyDisplay.classList.add('good');
            } else if (accuracy >= 70) {
                accuracyDisplay.classList.add('warning');
            } else {
                accuracyDisplay.classList.add('bad');
            }
        } else {
            feedbackText.innerHTML = '<div class="feedback-item">No pose detected. Please step into frame.</div>';
            accuracyDisplay.textContent = "0%";
            accuracyDisplay.className = 'bad';
        }
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', async () => {
    videoElement = document.getElementById('webcam');
    canvasElement = document.getElementById('canvas');
    canvasCtx = canvasElement.getContext('2d');

    // Initialize pose landmarker
    feedbackText.innerHTML = '<div class="feedback-item">Loading AI model...</div>';
    await initializePoseLandmarker();
    feedbackText.innerHTML = '<div class="feedback-item">Ready! Please calibrate first.</div>';
    calibrateButton.disabled = false;

    // Event listeners
    startButton.addEventListener('click', enableCam);
    calibrateButton.addEventListener('click', calibrate);
    poseSelect.addEventListener('change', (e) => {
        selectedPose = e.target.value;
        feedbackText.innerHTML = `<div class="feedback-item">Selected: ${referencePoses[selectedPose].name}</div>
                                  <div class="feedback-item">${referencePoses[selectedPose].description}</div>`;
    });
});