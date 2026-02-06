# Face Recognition using Linear Regression Classification (LRC)
**Author:** Arman Azadi
**Institution:** Iran University of Science and Technology (IUST)

## 📌 Project Overview
This project implements **Linear Regression Classification (LRC)** for face recognition. The core idea is that images of the same subject lie on a linear subspace. Given a test image $y$, the algorithm represents it as a linear combination of the training images from each class. The class that reconstructs $y$ with the minimum error is predicted as the identity.

The model is evaluated on the **AT&T (ORL) Face Database** (39 subjects, 10 images each).

## 🔬 Methodology
The system approaches classification as a regression problem:

### 1. Mathematical Formulation
For each class $i$, we construct a matrix $X_i$ containing its training images as column vectors. We solve for the coefficient vector $\beta_i$ that minimizes the reconstruction error:

$$\hat{\beta}_i = (X_i^T X_i + \lambda I)^{-1} X_i^T y$$

Where:
* $X_i$: Training images for class $i$.
* $y$: The test image vector.
* $\lambda$: Regularization parameter (Tikhonov Regularization) to handle collinearity.

The distance (error) for each class is calculated as:
$$d_i = || y - X_i \hat{\beta}_i ||_2$$
The predicted class is $\arg \min_i (d_i)$.

### 2. Robustness Testing (Downsampling)
To evaluate the model's robustness to low-resolution data, a second experiment (`Robustness_Test_Downsampling.ipynb`) was conducted where all images were downsampled by 50% (reducing the feature vector size by 75%).

## 📊 Dataset
* **Source:** `att_faces.mat` (AT&T/ORL Database)
* **Structure:** 39 Classes, 10 images per class.
* **Dimensions:** Original 112x92 pixels (vector size: 10,304).
* **Split:** 7 images for training, 3 for testing (Randomized).

## 📈 Results
* **Full Resolution:** The LRC model achieves high accuracy by effectively projecting test faces onto the subject-specific subspaces.
* **Regularization:** The report indicates that adding regularization ($\lambda=1$) provided stable results similar to the non-regularized version ($\lambda=0$).
* **Low Resolution:** The model maintains acceptable recognition rates even when image dimensions are halved, demonstrating the robustness of the subspace projection method.

*(See `Technical_Report.pdf` for detailed performance analysis.)*

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Linear Algebra:** `NumPy` (Matrix inversion, Dot products)
* **Data Handling:** `SciPy` (Loading .mat files)
* **Visualization:** `Matplotlib`

## 🚀 Usage
1.  Clone the repository:
    ```bash
    git clone [https://github.com/armanazadi/face-recognition-linear-regression.git](https://github.com/armanazadi/face-recognition-linear-regression.git)
    ```
2.  Run the main notebook:
    ```bash
    jupyter notebook Face_Recognition_LRC.ipynb
    ```
