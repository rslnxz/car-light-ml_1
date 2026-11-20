# CarID: Identifying Car Models from Headlight Photos

CarID is an image classifier that predicts a car's make, model and generation from a close-up photo of its headlight. It was trained on 1,540 headlight photos of 62 car models, all taken by me, and is deployed as a web app.

**Live app:** https://car-id.streamlit.app

> **Status: research prototype.** It works well on close-up phone photos similar to its training data and much worse on screenshots and internet images. See [Limitations](#limitations).

## Why headlights

Car makers design headlights as a brand signature. The shape of the LED strips, the daytime running lights and the housing usually changes with each model generation. This project tests whether a convolutional neural network can tell car models apart using only that one detail.

## Features

- Upload a photo, or take one with the camera, and get the most likely car model with a confidence score, plus the top 8 predictions.
- A warning when the top prediction is below 60% confidence.
- **Related vehicles:** for each predicted model, a hand-written lookup table lists trims and closely related models that use the same headlight design. This expands the 62 trained classes to 321 vehicles. These are not separate model predictions.
- **Analysis tab:** pipeline overview, per-class test accuracy, project history, limitations and future work.

## How it works

```
Photo → resize to 224×224 → ResNet18 (pretrained on ImageNet) → 62 class scores → softmax probabilities → related-vehicle lookup
```

| Stage | Details |
|---|---|
| Data | 1,540 photos of 62 car models, about 20–30 per model. Taken with an iPhone at a car dealership and in parking lots, square crop, headlight filling most of the frame. One folder per model. |
| Preprocessing | Resize to 224×224 and normalize with ImageNet mean and standard deviation. Training augmentation: random horizontal flip. |
| Model | torchvision ResNet18 pretrained on ImageNet. The final fully connected layer is replaced with a 62-class layer, and all layers are fine-tuned (transfer learning). |
| Training | Adam optimizer (learning rate 0.001), cross-entropy loss, batch size 4, 10 epochs, in Google Colab. The weights from the epoch with the lowest validation loss are kept. |
| Evaluation | Random 70/15/15 split with a fixed seed: 1,078 training, 231 validation, 231 test photos. |
| Deployment | Streamlit Community Cloud. The ~45 MB weights file is stored on Google Drive and downloaded with gdown when the app first starts. |

## Results

| | Proof of concept (July 2026) | Current version (November 2026) |
|---|---|---|
| Car models | 5 | 62 |
| Photos | 98 | 1,540 |
| Test photos | about 20 | 231 |
| Models tried | custom 3-layer CNN, ResNet18 | ResNet18 |
| Test accuracy | 90% (custom CNN), 100% (ResNet18) | 96.1% (94.8% in an earlier training run) |

- The most common error is BMW X5 photos predicted as Toyota 4Runner (3 of the 4 BMW X5 test photos). The same confusion already appeared in the 5-model proof of concept.
- With about 4 test photos per model, per-class accuracy is noisy: one photo changes a model's score by about 25 percentage points.
- In the proof of concept, both models scored highly on my own photos but failed on a photo from the internet. That gap is still the main open problem (see below).

## Limitations

The test accuracy above should not be read as real-world accuracy.

1. **Optimistic test split.** Photos were split at random, and most models were photographed as one car in one session. Nearly identical shots of the same car end up in both the training and test sets, so the test mostly checks photos the model has effectively already seen.
2. **One car per model.** With a single example car, the model can learn that car's paint color, reflections and surroundings instead of the headlight design shared by every car of that model.
3. **Screenshots and internet photos.** These usually show the whole car, with studio lighting, editing and compression. After resizing to 224×224, the headlight covers only a small part of the image. Testing on screenshots gave much lower accuracy, and changing training settings did not fix it, which points to the data rather than the model settings.
4. **Closed set.** The model always picks one of its 62 models, so a car outside the list gets a wrong label, sometimes with high confidence.
5. **Little augmentation.** Training used only horizontal flips, so the model saw almost no variation in zoom, lighting or angle.
6. **Small dataset.** About 25 photos per model is very little for a 62-class image problem.
7. **The related-vehicle list is manual.** It assumes trims share the same headlight unit; some trims and model years use different headlights.

## Future work

- **More varied data:** photograph several cars per model in different colors, lighting (overcast, night) and angles, and add internet photos to training.
- **Honest evaluation:** a fixed real-world test set of screenshots and internet photos that is never used for training, and train/test splits by car or photo session instead of by individual photo.
- **Stronger augmentation:** random crops and zoom, brightness and color changes, blur, perspective shifts.
- **Training changes:** a lower learning rate for fine-tuning, a learning-rate schedule, label smoothing.
- **More advanced models:** compare EfficientNet, ConvNeXt and Vision Transformers against ResNet18.
- **Two-stage pipeline:** first find and crop the headlight with an object detector (for example YOLO), then classify the crop, so full-car photos work.
- **Explainability:** Grad-CAM heatmaps to check whether the model looks at the headlight or at the background.
- **Unknown cars:** a confidence threshold or open-set method so the app can answer "not in the database".

## Tech stack

Python 3.11 · PyTorch · torchvision · Streamlit · Pillow · gdown · Google Colab · Google Drive

## Repository contents

| File | Purpose |
|---|---|
| `app.py` | Streamlit web app: model loading, prediction, interface, analysis tab |
| `requirements.txt` | Python dependencies |
| `class_names.txt` | The 62 class labels, in the model's output order |

Training was done in Google Colab. The dataset (1,540 photos) and the model weights are stored on Google Drive and are not included in this repository.

## Run locally

```bash
git clone https://github.com/rslnxz/car-light-ml_1.git
cd car-light-ml_1
pip install -r requirements.txt
streamlit run app.py
```

The first launch downloads the model weights (~45 MB).

## Supported car models (62)

Years are the year of the photographed car. A headlight design is usually shared across a model generation.

<details>
<summary>Show the full list</summary>

- Acura MDX (2026)
- Audi Q8 (2021)
- BMW iX M60 (2023)
- BMW M3 (2025)
- BMW X3 (2017)
- BMW X4
- BMW X5 40i (2021)
- BMW X7 (2022)
- BMW Z4 (2013)
- Buick Envista
- Cadillac Escalade (2024)
- Cadillac XT4 (2021)
- Chevrolet Tahoe (2024)
- Chevrolet Traverse (2024)
- Dodge Challenger
- Ford Escape, 3rd gen (2013–2016)
- Ford Expedition (2021)
- Ford Explorer (2022)
- Genesis GV80
- GMC Yukon
- Honda Civic Si (2024)
- Honda CR-V, 6th gen (2023–2025)
- Honda HR-V Sport
- Honda Odyssey
- Hyundai Elantra (2023)
- Hyundai Palisade (2022)
- Hyundai Santa Fe
- Infiniti QX50 (2019)
- Infiniti QX60 (2023)
- Infiniti QX80
- Jeep Grand Cherokee WL, 5th gen (2022)
- Jeep Wrangler Rubicon (2021)
- Kia K5 GT (2026)
- Kia Optima
- Kia Sorento
- Kia Sportage
- Land Rover Defender 110 (2023)
- Lexus ES 350 (2025)
- Lexus IS 250 (2014)
- Lexus RX
- Lexus UX 300h (2025)
- Mercedes-AMG A35
- Mercedes-Benz C 300 (2024)
- Mercedes-Benz EQE 350 (2023)
- Mercedes-Benz G 550 (2017)
- Mercedes-Benz GLA 250 (2021)
- Mercedes-Benz GLE 350 (2024)
- Mitsubishi Lancer GT
- Nissan Titan SV
- Porsche Panamera
- Subaru Outback (2025)
- Toyota 4Runner, 5th gen (2010–2024)
- Toyota Avalon (2014)
- Toyota Corolla (2022)
- Toyota Highlander
- Toyota RAV4 (2024)
- Toyota Tundra (2024)
- Toyota Venza (2022)
- Volkswagen Atlas (2023)
- Volkswagen Atlas Cross Sport (2021)
- Volkswagen Tiguan (2025)
- Volvo XC90 (2020)

</details>

## How this project was built

I chose the problem, photographed and labeled the dataset, ran training and evaluation in Google Colab, analyzed the errors, and deployed the app. I started with no coding experience. I used an AI assistant (Claude) to help write the code, and I used it as a tutor so I could learn how the code actually works (tensors, convolutional networks, transfer learning, training loops, validation, overfitting), not just to have the work done for me. A mentor guided the project, reviewed the results and ran additional experiments.

## Author

Rus Surmin
