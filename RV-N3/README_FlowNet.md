# FlowNet Implementation - Optični Pretok

## Pregled implementacije

Ta projekt implementira **FlowNetSimple** arhitekturo za napovedovanje optičnega pretoka, kot je opisano v članku "FlowNet: Learning Optical Flow with Convolutional Networks".

## Izvedene naloge

### ✅ 1. Priprava podatkov (15 točk)
- **Nalaganje optičnega pretoka (5 točk)**: Implementiran razred `FlyingChairsOfficial` za nalaganje Flying Chairs dataseta
- **Vizualizacija optičnega pretoka (10 točk)**: HSV vizualizacija, kjer:
  - **Hue (odtenek)**: smer optičnega pretoka (kot)
  - **Saturation (nasičenost)**: moč optičnega pretoka (magnituda)
  - **Value (svetlost)**: nastavljena na 1 za maksimalen kontrast

### ✅ 2. FlowNetSimple arhitektura

#### Encoder (kontrakcijski del):
- **conv1**: 7×7 konvolucija, 64 filtrov, stride 2
- **conv2**: 5×5 konvolucija, 128 filtrov, stride 2
- **conv3, conv3_1**: 5×5 in 3×3 konvolucije, 256 filtrov
- **conv4, conv4_1**: 3×3 konvolucije, 512 filtrov
- **conv5, conv5_1**: 3×3 konvolucije, 512 filtrov
- **conv6, conv6_1**: 3×3 konvolucije, 1024 filtrov

#### Decoder (ekspanzijski del):
- **Dekonvolucijske plasti** za upsampling
- **Skip povezave** med encoder in decoder plastmi
- **Multi-scale flow napovedi** na različnih resolucijah (flow2, flow3, flow4, flow5, flow6)

#### Aktivacijske funkcije:
- **LeakyReLU(0.1)** v vseh konvolucijskih plasteh

### ✅ 3. Funkcija izgube (Loss Function)

Implementirana **MultiScaleEPE** (Endpoint Error) funkcija:
- Računa L2 razdaljo med napovedanim in pravim optical flow
- Multi-scale pristop z utežmi: `[0.005, 0.01, 0.02, 0.08, 0.32]`
- Prilagaja ground truth flow za vsako resolucijo

### ✅ 4. Učenje modela

#### Trening pipeline:
- **Optimizer**: Adam (learning rate = 1e-4)
- **Learning rate scheduler**: StepLR (zmanjšanje vsakih 100 epoch)
- **Batch size**: 8
- **Train/Val split**: uporaba FlyingChairs_train_val.txt

#### Funkcionalnosti:
- Progress bar z tqdm
- Shranjevanje checkpointov vsakih 5 epoch
- Sledenje train in validation loss

### ✅ 5. Vizualizacija rezultatov

Implementirane funkcije za:
- **Prikaz napovedi** modela (ground truth vs. predicted flow)
- **HSV vizualizacija** optičnega pretoka
- **Error mapa** (L2 razdalja med napovedan in resničnim flow)
- **EPE metrika** (Average Endpoint Error)
- **Training curves** (train/val loss)

## Uporaba

### 1. Nalaganje podatkov
```python
root = r"C:\Users\simon\Desktop\RV\Racunalniski Vid\RV-N3\datasets\FlyingChairs_release"
transform = transforms.ToTensor()
dataset = FlyingChairsOfficial(root=root, transform=transform, split="train")
```

### 2. Inicializacija modela
```python
model = FlowNetSimple(input_channels=6)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 3. Trening (odkomentiraj za zagon)
```python
train_losses, val_losses = train_flownet(
    model, 
    train_loader, 
    val_loader, 
    epochs=50, 
    lr=1e-4, 
    device=device
)
```

### 4. Vizualizacija napovedi
```python
visualize_flow_prediction(model, val_dataset, idx=0, device=device)
```

## Tehnične podrobnosti

### Model parametri:
- **Skupno število parametrov**: ~38M (prikaže se pri zagonu celice testiranja)
- **Input**: 6-kanalna slika (2× RGB slike)
- **Output**: 2-kanalni optical flow (u, v komponenti)

### Dataset:
- **Flying Chairs**: Sintetični dataset z optičnim pretokom
- **Format**: PPM slike, FLO optical flow
- **Split**: ~22k train, ~640 val samples

## Možne izboljšave

1. **Data augmentation**: Random crop, rotation, color jitter
2. **FlowNetCorr**: Implementacija korelacijske verzije
3. **Fine-tuning**: Nižji learning rate za konvergenco
4. **Batch normalization**: Stabilnejše učenje
5. **Predtrenirani model**: Uporaba predtreniranih uteži

## Reference

- **FlowNet paper**: Fischer et al., "FlowNet: Learning Optical Flow with Convolutional Networks", ICCV 2015
- **Dataset**: Flying Chairs dataset
- **Framework**: PyTorch

