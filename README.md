# TagRuler: Interactive Tool for Span-Level Data Programming by Demonstration
This repo contains the source code and the user evaluation data for TagRuler, a data programming by demonstration system for span-level annotation.
Check out our [demo video](https://youtu.be/MRc2elPaZKs) to see TagRuler in action!

# <a name='Use'></a>How to use the source code in this repo

Follow these instructions to run the system on your own, where you can plug in your own data and save the resulting labels, models, and annotations.

## 1. Server

### 1-1. Install Dependencies :wrench:

```shell
cd server
pip install -r requirements.txt
```

### 1-2. (Optional) Download Data Files

- **Hotel** ([Download](https://drive.google.com/file/d/1tDsFvGFUg_QvUp4tsIi3TfirHNSnTdH4/view?usp=sharing)): Hotel reviews for Aspect-Opinion annotation
- **Restaurant** ([Download](https://drive.google.com/file/d/1Avu0aMfvqEKn9DOon2dTUIvwjy0Hownh/view?usp=sharing)): Restaurant reviews for Aspect-Opinion annotation
- **BC5CDR** ([Download](https://drive.google.com/file/d/1kKeINUOjtCVGr1_L3aC3qDo3-O-jr5hR/view?usp=sharing)): PubMed articles for Chemical-Disease annotation

### 1-3. Run :runner:

```
python api/server.py
```

## 2. User Interface

### 2-1. Install Node.js

[You can download node.js here.](https://nodejs.org/en/)

To confirm that you have node.js installed, run `node - v`

### 2-2. Run

```shell
cd ui
npm install 
npm start
```

By default, the app will make calls to `localhost:5000`, assuming that you have the server running on your machine. (See the [instructions above](#Engine)).

Once you have both of these running, navigate to `localhost:3000`.
