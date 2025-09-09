# Plant Disease Detection Frontend

A modern, responsive web interface for the Plant Disease Detection API.

## Features

- **Drag & Drop Upload**: Easy image upload with drag and drop support
- **Real-time Prediction**: Instant disease detection with confidence scores
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Image Preview**: Shows uploaded image with prediction results
- **Quality Analysis**: Displays image quality metrics
- **Multiple Predictions**: Shows top predictions with confidence scores

## Supported Plant Diseases

### Tomato
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Yellow Leaf Curl Virus
- Mosaic Virus
- Healthy

### Potato
- Early Blight
- Late Blight
- Healthy

### Pepper
- Bacterial Spot
- Healthy

## Usage

### Option 1: Direct HTML
1. Open `index.html` in a web browser
2. Make sure the API server is running on `http://localhost:8000`
3. Upload an image to get predictions

### Option 2: Local Server
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Start a local server:
   ```bash
   python -m http.server 3000
   ```

3. Open your browser and go to `http://localhost:3000`

### Option 3: With Node.js (if available)
1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

## API Configuration

The frontend is configured to connect to the API at `http://localhost:8000`. To change this:

1. Open `index.html`
2. Find the line: `this.apiUrl = 'http://localhost:8000';`
3. Change it to your API server URL

## File Requirements

- **Supported Formats**: JPG, JPEG, PNG, GIF, BMP
- **Maximum Size**: 10MB
- **Recommended Resolution**: 224x224 pixels or higher
- **Image Quality**: Clear, well-lit images work best

## Browser Compatibility

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## Troubleshooting

### API Connection Issues
- Ensure the API server is running on the correct port
- Check that CORS is enabled in the API server
- Verify the API URL in the frontend code

### Image Upload Issues
- Check file format is supported
- Ensure file size is under 10MB
- Try with a different image

### Prediction Errors
- Check browser console for error messages
- Verify API server logs
- Ensure model is loaded correctly

## Development

### Customization
- Modify `index.html` to change the UI
- Update CSS styles for different themes
- Add new features to the JavaScript code

### API Integration
- The frontend uses the `/predict/base64` endpoint
- Modify the `predictDisease` method to use different endpoints
- Add support for additional API features

## License

MIT License - see LICENSE file for details.
