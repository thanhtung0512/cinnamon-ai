document.addEventListener('DOMContentLoaded', function() {
  const container = document.getElementById('container');
  const buttonContainer = document.createElement('div');
  buttonContainer.className = 'button-container';

  const getImagesButton = document.createElement('button');
  getImagesButton.className = 'button';
  getImagesButton.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M10 3.3c-4.6.4-8.3 4.3-8.3 9.1 0 4.7 3.7 8.5 8.3 8.9V23c0 .6.4 1 1 1s1-.4 1-1v-1.7c4.6-.4 8.3-4.3 8.3-9.1 0-4.7-3.7-8.5-8.3-8.9V1c0-.6-.4-1-1-1s-1 .4-1 1v1.3z"></path></svg> Get Images';

  const predictButton = document.createElement('button');
  predictButton.className = 'button';
  predictButton.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24"><path fill="currentColor" d="M12 23c-3.6 0-6.8-2.1-8.2-5.3-1.1-2.6-.5-5.6 1.4-7.8l-1.5-2.6C2.7 5.6 4 4.2 6 3.5c1.5-.5 3.1-.3 4.5.6L12 2c3.6 0 6.8 2.1 8.2 5.3 1.1 2.6.5 5.6-1.4 7.8l1.5 2.6C21.3 18.4 20 19.8 18 20.5c-1.5.5-3.1.3-4.5-.6l-1.3 2.1c-.3.5-.9.9-1.5.9zm0-16c-2.2 0-4 1.8-4 4s1.8 4 4 4 4-1.8 4-4-1.8-4-4-4z"></path></svg> Predict';

  buttonContainer.appendChild(getImagesButton);
  buttonContainer.appendChild(predictButton);
  container.appendChild(buttonContainer);

  const imageContainer = document.createElement('div');
  imageContainer.className = 'image-container';
  container.appendChild(imageContainer);

  getImagesButton.addEventListener('click', getRandomImages);
  predictButton.addEventListener('click', predictImages);

  let images = [];
  let predictions = [];
  let inferenceTime = [];
  let spin = false;
  async function getRandomImages() {
    spin = false;
    predictions = [];
    inferenceTime = [];
    const folderPaths = [
      '../dataset/OCR/Level3_samples/504',
      '../dataset/OCR/Level3_samples/503',
    ];

    const imagePaths = [];

    for (const folderPath of folderPaths) {
      const randomNumber = Math.floor(Math.random() * 27);
      const imagePath = `${folderPath}/${randomNumber}.jpg`;
      imagePaths.push(imagePath);
    }

    images = imagePaths;
    renderImages();
  }
async function predictImages() {
  // Clear previous predictions and inference times
  predictions = [];
  inferenceTime = [];
  spin = true; // Set spin to true before fetching predictions
  renderImages(); // Update UI to display spinner

  for (const imagePath of images) {
    const response = await fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image_path: imagePath }),
    });
    const data = await response.json();
    predictions.push(data.prediction);
    inferenceTime.push(data.inference_time);
  }

  spin = false; // Set spin to false after predictions are received
  renderImages(); // Update UI to hide spinner and display predictions
}


  function renderImages() {
  imageContainer.innerHTML = '';

  images.forEach((imagePath, index) => {
    const imageDiv = document.createElement('div');
    imageDiv.innerHTML = `
      <div class="prediction-container-wrapper">
        <img class="image" src="${imagePath}" alt="Image ${index + 1}">
        <div class="prediction-container">
          <p class="prediction">${ spin? "Predicting" :"Predict Result: " + predictions[index]}</p>
          <p class="inference-time">${inferenceTime[index] ? `Inference time: ${inferenceTime[index]} seconds` : ''}</p>
          <div class="loading-icon" style="display: ${spin ? 'block' : 'none'}">
            <div class="loader"></div>
          </div>
        </div>
      </div>
    `;
    imageContainer.appendChild(imageDiv);
  });
}

});
