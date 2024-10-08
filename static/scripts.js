// 1. Variable Declarations: Manage DOM references and state variables

const messagesDiv = document.getElementById('messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const newSessionButton = document.getElementById('new-session-button');
const modelSwitcher = document.getElementById('model-switcher');
const dropdownMenu = document.getElementById('dropdown-menu');
const dropdownButtons = dropdownMenu.querySelectorAll('button');
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebar = document.getElementById('sidebar');
const topBar = document.getElementById('top-bar');
const imageUploadButton = document.getElementById('image-upload-button');
const imageUploadInput = document.getElementById('image-upload');

let sessionActive = false;
let currentModel = "ChatGPT 4o mini";
let currentReader = null;
let isStreaming = false;
let base64Image = null;

// 2. Event Listener Setup: Define how the script interacts with user actions

// Toggle the sidebar visibility when the toggle button is clicked
sidebarToggle.addEventListener('click', toggleSidebar);

// Adjust the send button and textarea height based on user input
userInput.addEventListener('input', handleUserInput);
userInput.addEventListener('keydown', handleUserInputKeydown);

// Handle the send button click to send the user's message
sendButton.addEventListener('click', handleSendButtonClick);

// Handle new session button click to reset the chat session
newSessionButton.addEventListener('click', handleNewSessionButtonClick);

// Toggle the model dropdown menu visibility
modelSwitcher.addEventListener('click', toggleDropdownMenu);

// Update the current model based on the dropdown selection
dropdownButtons.forEach(button => {
    button.addEventListener('click', handleDropdownSelection);
});

// Close the model dropdown if a click happens outside of it
document.addEventListener('click', closeDropdownMenuOnClick);

// Handle clicks on the document to manage bot actions and audio streaming
document.addEventListener('click', handleDocumentClick);

// Trigger the file input when the upload button is clicked
imageUploadButton.addEventListener('click', triggerImageUpload);

// Handle image file selection and preview
imageUploadInput.addEventListener('change', handleImageSelection);

// Handle session clearance when page is refreshed
window.addEventListener('beforeunload', clearSessionOnUnload);

// 3. Utility Functions: Helper functions to perform specific tasks

/**
 * Display a message in the chat interface.
 * @param {string} text - The message text to display.
 * @param {string} sender - The sender of the message ('user' or 'bot').
 * @param {boolean} isLoading - Whether the message is a loading placeholder.
 * @returns {HTMLElement} The message element added to the DOM.
 */
function displayMessage(text, sender, isLoading = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    if (sender === 'bot') {
        const modelIconDiv = document.createElement('div');
        modelIconDiv.className = 'model-icon';

        let modelIconSrc;
        switch (currentModel.toLowerCase()) {
            case 'gemini pro':
                modelIconSrc = 'static/images/google.png';
                break;
            case 'chatgpt 4o mini':
                modelIconSrc = 'static/images/openai.png';
                break;
            case 'llama 3 (70b)':
                modelIconSrc = 'static/images/meta.png';
                break;
            case 'mixtral (8x7b)':
                modelIconSrc = 'static/images/mistral.png';
                break;
            default:
                modelIconSrc = ''; // Fallback if the model is not recognized
        }

        modelIconDiv.innerHTML = `<img src="${modelIconSrc}" alt="${currentModel} Icon" style="width: 100%; height: 100%; object-fit: cover;">`;
        if (isLoading) {
            modelIconDiv.classList.add('pulsing'); // Add pulsing animation for loading state
        }
        messageDiv.appendChild(modelIconDiv);

        const messageTextDiv = document.createElement('div');
        messageTextDiv.className = 'message-text';
        messageTextDiv.innerHTML = isLoading ? '' : window.marked.parse(text); // Display the message text
        messageDiv.appendChild(messageTextDiv);
    } else {
        messageDiv.innerHTML = window.marked.parse(text); // Display the user's message
    }

    messagesDiv.appendChild(messageDiv); // Add the message to the chat
    messagesDiv.scrollTop = messagesDiv.scrollHeight; // Scroll to the latest message

    // Activate new session button if there are messages
    if (messagesDiv.children.length > 0) {
        newSessionButton.classList.add('active');
        newSessionButton.disabled = false;
    }

    return messageDiv; // Return the created message element
}

/**
 * Add interactive buttons to bot messages.
 * @param {HTMLElement} messageDiv - The message element to which buttons will be added.
 * @param {string} text - The message text associated with the buttons.
 */
function addBotButtons(messageDiv, text) {
    const botButtonsContainer = document.createElement('div');
    botButtonsContainer.className = 'bot-buttons-container';

    const botButtonsDiv = document.createElement('div');
    botButtonsDiv.className = 'bot-buttons';

    const readButton = document.createElement('button');
    readButton.innerHTML = '<i class="fa-solid fa-volume-high"></i>';
    readButton.title = 'Lees voor'; // Read aloud button

    const likeButton = document.createElement('button');
    likeButton.innerHTML = '<i class="fa fa-thumbs-up"></i>';
    likeButton.title = 'Goede reactie'; // Like button

    const dislikeButton = document.createElement('button');
    dislikeButton.innerHTML = '<i class="fa fa-thumbs-down"></i>';
    dislikeButton.title = 'Slecte reactie'; // Dislike button

    const copyButton = document.createElement('button');
    copyButton.innerHTML = '<i class="fa fa-copy"></i>';
    copyButton.title = 'Kopiëren'; // Copy button

    // Handle copy button click to copy text to clipboard
    copyButton.addEventListener('click', () => {
        if (copyButton.querySelector('i').classList.contains('fa-copy')) {
            navigator.clipboard.writeText(text).then(() => {
                copyButton.innerHTML = '<i class="fa fa-check"></i>'; // Show check mark when copied
            });
        } else {
            copyButton.innerHTML = '<i class="fa fa-copy"></i>'; // Reset to copy icon
        }
    });

    botButtonsDiv.appendChild(readButton);
    botButtonsDiv.appendChild(likeButton);
    botButtonsDiv.appendChild(dislikeButton);
    botButtonsDiv.appendChild(copyButton);

    botButtonsContainer.appendChild(botButtonsDiv);
    messageDiv.appendChild(botButtonsContainer);
}

// 4. Main Functions: Core features and logic

// Function to handle toggling the sidebar
function toggleSidebar() {
    sidebar.classList.toggle('visible');
    if (sidebar.classList.contains('visible')) {
        document.body.style.marginLeft = '250px'; // Shift the body to the right
        topBar.classList.remove('shifted'); // Adjust the top bar
        sidebarToggle.classList.add('active'); // Indicate the sidebar is active
    } else {
        document.body.style.marginLeft = '0'; // Reset the body margin
        topBar.classList.add('shifted'); // Reapply the shifted class
        sidebarToggle.classList.remove('active'); // Indicate the sidebar is inactive
    }
}

// Function to handle user input
function handleUserInput() {
    sendButton.disabled = !userInput.value.trim(); // Disable send button if input is empty
    sendButton.classList.toggle('active', !sendButton.disabled); // Style the button if it's active
    userInput.style.height = 'auto'; // Reset height to adjust dynamically
    userInput.style.height = `${userInput.scrollHeight}px`; // Adjust height to fit content
}

// Function to handle keydown events in the user input
function handleUserInputKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault(); // Prevent newline from being added
        sendButton.click(); // Simulate a click on the send button
    }
}

// Function to handle sending the user's message
async function handleSendButtonClick() {
    const userMessage = userInput.value.trim(); // Get the user's message
    if (userMessage || base64Image) { // Check if there is a message or an image
        document.getElementById('placeholder-message').style.display = 'none'; // Hide placeholder message
        document.getElementById('svg-container').style.display = 'none'; // Hide SVG container

        displayMessage(userMessage, 'user'); // Display user's message
        userInput.value = ''; // Clear the input
        sendButton.disabled = true; // Disable the send button
        userInput.style.height = 'auto'; // Reset input height

        const loadingMessageDiv = displayMessage("", 'bot', true); // Show loading message

        // Send the message to the server
        const response = await fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                message: userMessage, 
                model: currentModel, 
                session: sessionActive,
                image: base64Image
            }),
        });

        const result = await response.json(); // Parse the response

        // Update the bot message with the server response
        const messageTextDiv = loadingMessageDiv.querySelector('.message-text');
        if (messageTextDiv) {
            messageTextDiv.innerHTML = window.marked.parse(result.response); // Parse and display the response
        }

        const modelIconDiv = loadingMessageDiv.querySelector('.model-icon');
        if (modelIconDiv) {
            modelIconDiv.classList.remove('pulsing'); // Stop pulsing animation
        }

        addBotButtons(loadingMessageDiv, result.response); // Add interaction buttons to the message

        // Enable new session button if there are messages
        if (messagesDiv.children.length > 0) {
            newSessionButton.classList.add('active');
            newSessionButton.disabled = false;
        }

        base64Image = null; // Clear the image data
    }
}

// Function to handle starting a new session
async function handleNewSessionButtonClick() {
    if (messagesDiv.children.length > 0) {
        await fetch('/new_session', { method: 'POST', headers: { 'Content-Type': 'application/json' } });
        messagesDiv.innerHTML = ''; // Clear the messages
        sessionActive = false; // Reset session state
        newSessionButton.classList.remove('active'); // Deactivate the button
        newSessionButton.disabled = true; // Disable the button
    }
}

// Function to delete session file when page is refreshed
function clearSessionOnUnload() {
    navigator.sendBeacon('/clear-session');
}

// Function to toggle the model dropdown menu
function toggleDropdownMenu() {
    dropdownMenu.style.display = dropdownMenu.style.display === 'block' ? 'none' : 'block';
}

// Function to handle dropdown menu selection
function handleDropdownSelection(event) {
    const button = event.target.closest('button');
    if (button && dropdownMenu.contains(button)) {
        const model = button.getAttribute('data-model'); // Update the current model
        currentModel = model;
        modelSwitcher.innerHTML = `${currentModel} <i class="fas fa-chevron-down"></i>`; // Update the button text
        dropdownButtons.forEach(btn => btn.classList.remove('selected')); 
        button.classList.add('selected'); // Select the clicked button
        dropdownMenu.style.display = 'none'; // Hide the dropdown menu
    }
}

// Function to close dropdown menu if click happens outside of it
function closeDropdownMenuOnClick(e) {
    if (!modelSwitcher.contains(e.target)) {
        dropdownMenu.style.display = 'none'; // Hide the dropdown menu
    }
}

// Function to handle document clicks for managing bot actions and audio streaming
function handleDocumentClick(event) {
    const target = event.target;

    // Check if the click is on a read aloud or stop button
    if (target.closest('.bot-buttons button i.fa-volume-high') || target.closest('.bot-buttons button i.fa-stop')) {
        const readButton = target.closest('.bot-buttons button');
        const icon = readButton.querySelector('i');
        const messageText = readButton.closest('.message.bot').querySelector('.message-text').innerText;

        if (icon.classList.contains('fa-volume-high')) {
            handleAudioStream(icon, messageText);
        } else if (icon.classList.contains('fa-stop')) {
            stopAudioStream(icon);
        }
    }
}

// Function to handle audio streaming when read aloud button is clicked
function handleAudioStream(icon, messageText) {
    // Stop any current streaming
    if (isStreaming) {
        if (currentReader) {
            currentReader.cancel(); // Cancel the current stream
        }
        isStreaming = false;
        currentReader = null;
        document.querySelectorAll('.bot-buttons i.fa-stop').forEach(stopIcon => {
            stopIcon.className = 'fa fa-volume-high'; // Reset icon to volume high
        });
    }

    icon.className = 'loading-icon'; // Show loading icon

    // Send request to read aloud the message
    fetch('/read_aloud', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: messageText })
    })
    .then(response => {
        const reader = response.body.getReader();
        currentReader = reader;
        isStreaming = true;
        let streamStarted = false;

        // Read the stream and handle audio playback
        function readStream() {
            return reader.read().then(({ done, value }) => {
                if (done) {
                    icon.className = 'fa fa-volume-high'; // Reset icon when done
                    isStreaming = false;
                    currentReader = null;
                    return;
                }

                const textChunk = new TextDecoder().decode(value);
                if (!streamStarted) {
                    try {
                        const parsedData = JSON.parse(textChunk.trim());
                        if (parsedData.status === "stream_started") {
                            icon.className = 'fa fa-stop'; // Change to stop icon
                            streamStarted = true;
                        }
                    } catch (e) {
                        // Handle JSON parse errors
                    }
                }

                readStream(); // Continue reading stream
            });
        }

        readStream(); // Start reading stream
    })
    .catch(error => {
        console.error('Error:', error);
        icon.className = 'fa fa-volume-high'; // Reset icon on error
        isStreaming = false;
        currentReader = null;
    });
}

// Function to stop audio streaming
function stopAudioStream(icon) {
    if (isStreaming && currentReader) {
        currentReader.cancel(); // Cancel the current stream
        fetch('/stop_audio', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        }).then(() => {
            icon.className = 'fa fa-volume-high'; // Reset icon
            isStreaming = false;
            currentReader = null;
        });
    }
}

// Function to trigger image upload
function triggerImageUpload() {
    imageUploadInput.click();
}

// Function to handle image selection and preview
function handleImageSelection() {
    const file = imageUploadInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onloadend = () => {
            base64Image = reader.result.split(',')[1]; // Convert image to base64
            
            const imgElement = document.createElement('img');
            imgElement.src = `data:image/jpeg;base64,${base64Image}`; // Display base64 image
            
            const imageContainerDiv = document.getElementById('image-container');
            imageContainerDiv.innerHTML = ''; // Clear previous content
            imageContainerDiv.appendChild(imgElement); // Add new image
            
            const deleteIcon = document.createElement('button');
            deleteIcon.className = 'icon-button delete-icon';
            deleteIcon.innerHTML = '<i class="fas fa-times"></i>'; // Add delete icon
            
            imageContainerDiv.appendChild(deleteIcon);
            document.getElementById('image-preview').style.display = 'block'; // Show the preview

            // Allow the user to remove the uploaded image
            deleteIcon.addEventListener('click', () => {
                imageContainerDiv.innerHTML = ''; // Clear the image container
                document.getElementById('image-preview').style.display = 'none'; // Hide the preview
                base64Image = null; // Clear the base64 image data
                imageUploadInput.value = ''; // Reset the file input
            });
        };
        reader.readAsDataURL(file); // Read the file as a data URL
    }
}
