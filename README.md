# TagRuler: Interactive Tool for Span-Level Data Programming by Demonstration
This repo contains the source code and the user evaluation data for TagRuler, a data programming by demonstration system for span-level annotation.
Check out our [demo video](https://youtu.be/MRc2elPaZKs) to see TagRuler in action!


<h3 align="center">
TagRuler synthesizes labeling functions based on your annotations, allowing you to quickly and easily generate large amounts of training data for span annotation, without the need to program. <br/>
 <a href="https://youtu.be/MRc2elPaZKs"><img width=800px src=tagruler-teaser.gif></a>
</h3>

### Demo Video!

https://youtu.be/MRc2elPaZKs

# <a name='About'></a>What is TagRuler?

In 2020, we introduced [Ruler](https://github.com/megagonlabs/ruler), a novel data programming by demonstration system that allows domain experts to leverage data programming without the need for coding.  Ruler generates document classification rules, but we knew that there was a bigger challenge left to tackle:  span-level annotations. This is one of the more time-consuming labelling tasks, and creating a DPBD system for this proved to be a challenge because of the sheer magnitude of the space of labeling functions over spans.

We feel that this is a critical extension of the DPBD paradigm, and that by open-sourcing it, we can help with all kinds of labelling needs.

# <a name='Use'></a>How to use the source code in this repo

Follow these instructions to run the system on your own, where you can plug in your own data and save the resulting labels, models, and annotations.

## 1. Server

### 1-1. Install Dependencies :wrench:

```shell
cd server
pip install -r requirements.txt
```

### 1-2. (Optional) Download Data Files

- **BC5CDR** ([Download Preprocessed Data](https://drive.google.com/file/d/1kKeINUOjtCVGr1_L3aC3qDo3-O-jr5hR/view?usp=sharing)): PubMed articles for Chemical-Disease annotation
Li, Jiao & Sun, Yueping & Johnson, Robin & Sciaky, Daniela & Wei, Chih-Hsuan & Leaman, Robert & Davis, Allan Peter & Mattingly, Carolyn & Wiegers, Thomas & lu, Zhiyong. (2016). Original database URL: http://www.biocreative.org/tasks/biocreative-v/track-3-cdr/

- **Your Own Data** See instructions in [server/datasets](server/datasets)

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
