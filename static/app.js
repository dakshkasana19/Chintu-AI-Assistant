document.addEventListener('DOMContentLoaded', () => {
    const setupButton = document.querySelector('.bottom-menu .menu-btn:first-child');
    const modal = document.getElementById('setup-modal');
    const closeModalBtn = document.getElementById('close-modal-btn');
    const registerBtn = document.getElementById('register-btn');
    const appPathInput = document.getElementById('app-path');
    const appNameInput = document.getElementById('app-name');
    const setupMessage = document.getElementById('setup-message');

    // Functions to open/close the modal
    setupButton.addEventListener('click', () => {
        modal.style.display = 'flex';
        setupMessage.textContent = ''; // Clear previous messages
        appPathInput.value = '';
        appNameInput.value = '';
    });

    closeModalBtn.addEventListener('click', () => {
        modal.style.display = 'none';
    });

    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });

    // --- Registration Logic: Sends form data to Flask server ---
    registerBtn.addEventListener('click', async () => {
        const path = appPathInput.value;
        const name = appNameInput.value;

        if (!path || !name) {
            setupMessage.textContent = "Please fill out both fields.";
            return;
        }

        setupMessage.textContent = "Registering...";

        try {
            const response = await fetch('/register_app', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ path: path, name: name })
            });

            const result = await response.json();

            if (result.success) {
                setupMessage.style.color = '#7CFC00'; // Green
                setupMessage.textContent = result.message;
                // Clear inputs on success
                appPathInput.value = '';
                appNameInput.value = '';
            } else {
                setupMessage.style.color = '#FF4500'; // Red/Orange
                setupMessage.textContent = `Failed: ${result.message}`;
            }

        } catch (error) {
            setupMessage.style.color = '#FF4500';
            setupMessage.textContent = `Communication error: Server might be down.`;
        }
    });

});