# Hugging Face Inference Endpoints

### Using in html
instead of readFileSync place file blob or url to data

### Initialize

```typescript
import { HfInference } from "https://esm.sh/@huggingface/inference"

const hf = new HfInference('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')
```

## Natural Language Processing

### Text Generation

Generates text from an input prompt.

```typescript
await hf.textGeneration({
  model: 'gpt2',
  inputs: 'The answer to the universe is'
})

for await (const output of hf.textGenerationStream({
  model: "google/flan-t5-xxl",
  inputs: 'repeat "one two three four"',
  parameters: { max_new_tokens: 250 }
})) {
  console.log(output.token.text, output.generated_text);
}
```

### Text Generation (Chat Completion API Compatible)

Using the `chatCompletion` method, you can generate text with models compatible with the OpenAI Chat Completion API. 

```typescript
// Non-streaming API
const out = await hf.chatCompletion({
  model: "mistralai/Mistral-7B-Instruct-v0.2",
  messages: [{ role: "user", content: "Complete the this sentence with words one plus one is equal " }],
  max_tokens: 500,
  temperature: 0.1,
  seed: 0,
});

// Streaming API
let out = "";
for await (const chunk of hf.chatCompletionStream({
  model: "mistralai/Mistral-7B-Instruct-v0.2",
  messages: [
    { role: "user", content: "Complete the equation 1+1= ,just the answer" },
  ],
  max_tokens: 500,
  temperature: 0.1,
  seed: 0,
})) {
  if (chunk.choices && chunk.choices.length > 0) {
    out += chunk.choices[0].delta.content;
  }
}
```

### Fill Mask

Tries to fill in a hole with a missing word (token to be precise).

```typescript
await hf.fillMask({
  model: 'bert-base-uncased',
  inputs: '[MASK] world!'
})
```

### Summarization

Summarizes longer text into shorter text. Be careful, some models have a maximum length of input.

```typescript
await hf.summarization({
  model: 'facebook/bart-large-cnn',
  inputs:
    'The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930.',
  parameters: {
    max_length: 100
  }
})
```

### Question Answering

Answers questions based on the context you provide.

```typescript
await hf.questionAnswering({
  model: 'deepset/roberta-base-squad2',
  inputs: {
    question: 'What is the capital of France?',
    context: 'The capital of France is Paris.'
  }
})
```

### Table Question Answering

```typescript
await hf.tableQuestionAnswering({
  model: 'google/tapas-base-finetuned-wtq',
  inputs: {
    query: 'How many stars does the transformers repository have?',
    table: {
      Repository: ['Transformers', 'Datasets', 'Tokenizers'],
      Stars: ['36542', '4512', '3934'],
      Contributors: ['651', '77', '34'],
      'Programming language': ['Python', 'Python', 'Rust, Python and NodeJS']
    }
  }
})
```

### Text Classification

Often used for sentiment analysis, this method will assign labels to the given text along with a probability score of that label.

```typescript
await hf.textClassification({
  model: 'distilbert-base-uncased-finetuned-sst-2-english',
  inputs: 'I like you. I love you.'
})
```

### Token Classification

Used for sentence parsing, either grammatical, or Named Entity Recognition (NER) to understand keywords contained within text.

```typescript
await hf.tokenClassification({
  model: 'dbmdz/bert-large-cased-finetuned-conll03-english',
  inputs: 'My name is Sarah Jessica Parker but you can call me Jessica'
})
```

### Translation

Converts text from one language to another.

```typescript
await hf.translation({
  model: 't5-base',
  inputs: 'My name is Wolfgang and I live in Berlin'
})

await hf.translation({
  model: 'facebook/mbart-large-50-many-to-many-mmt',
  inputs: textToTranslate,
  parameters: {
  "src_lang": "en_XX",
  "tgt_lang": "fr_XX"
 }
})
```

### Zero-Shot Classification

Checks how well an input text fits into a set of labels you provide.

```typescript
await hf.zeroShotClassification({
  model: 'facebook/bart-large-mnli',
  inputs: [
    'Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!'
  ],
  parameters: { candidate_labels: ['refund', 'legal', 'faq'] }
})
```

### Conversational

This task corresponds to any chatbot-like structure. Models tend to have shorter max_length, so please check with caution when using a given model if you need long-range dependency or not.

```typescript
await hf.conversational({
  model: 'microsoft/DialoGPT-large',
  inputs: {
    past_user_inputs: ['Which movie is the best ?'],
    generated_responses: ['It is Die Hard for sure.'],
    text: 'Can you explain why ?'
  }
})
```

### Sentence Similarity

Calculate the semantic similarity between one text and a list of other sentences.

```typescript
await hf.sentenceSimilarity({
  model: 'sentence-transformers/paraphrase-xlm-r-multilingual-v1',
  inputs: {
    source_sentence: 'That is a happy person',
    sentences: [
      'That is a happy dog',
      'That is a very happy person',
      'Today is a sunny day'
    ]
  }
})
```

## Audio

### Automatic Speech Recognition

Transcribes speech from an audio file.


```typescript
await hf.automaticSpeechRecognition({
  model: 'facebook/wav2vec2-large-960h-lv60-self',
  data: readFileSync('test/sample1.flac')
})
```

### Audio Classification

Assigns labels to the given audio along with a probability score of that label.


```typescript
await hf.audioClassification({
  model: 'superb/hubert-large-superb-er',
  data: readFileSync('test/sample1.flac')
})
```

### Text To Speech

Generates natural-sounding speech from text input.

```typescript
await hf.textToSpeech({
  model: 'espnet/kan-bayashi_ljspeech_vits',
  inputs: 'Hello world!'
})
```

### Audio To Audio

Outputs one or multiple generated audios from an input audio, commonly used for speech enhancement and source separation.

```typescript
await hf.audioToAudio({
  model: 'speechbrain/sepformer-wham',
  data: readFileSync('test/sample1.flac')
})
```

## Computer Vision

### Image Classification

Assigns labels to a given image along with a probability score of that label.

```typescript
await hf.imageClassification({
  data: readFileSync('test/cheetah.png'),
  model: 'google/vit-base-patch16-224'
})
```

### Object Detection

Detects objects within an image and returns labels with corresponding bounding boxes and probability scores.

```typescript
await hf.objectDetection({
  data: readFileSync('test/cats.png'),
  model: 'facebook/detr-resnet-50'
})
```

### Image Segmentation

Detects segments within an image and returns labels with corresponding bounding boxes and probability scores.

```typescript
await hf.imageSegmentation({
  data: readFileSync('test/cats.png'),
  model: 'facebook/detr-resnet-50-panoptic'
})
```

### Image To Text

Outputs text from a given image, commonly used for captioning or optical character recognition.

```typescript
await hf.imageToText({
  data: readFileSync('test/cats.png'),
  model: 'nlpconnect/vit-gpt2-image-captioning'
})
```

### Text To Image

Creates an image from a text prompt.

```typescript
await hf.textToImage({
  inputs: 'award winning high resolution photo of a giant tortoise/((ladybird)) hybrid, [trending on artstation]',
  model: 'stabilityai/stable-diffusion-2',
  parameters: {
    negative_prompt: 'blurry',
  }
})
```

### Image To Image

Image-to-image is the task of transforming a source image to match the characteristics of a target image or a target image domain.


```typescript
await hf.imageToImage({
  inputs: new Blob([readFileSync("test/stormtrooper_depth.png")]),
  parameters: {
    prompt: "elmo's lecture",
  },
  model: "lllyasviel/sd-controlnet-depth",
});
```

### Zero Shot Image Classification

Checks how well an input image fits into a set of labels you provide.

```typescript
await hf.zeroShotImageClassification({
  model: 'openai/clip-vit-large-patch14-336',
  inputs: {
    image: await (await fetch('https://placekitten.com/300/300')).blob()
  },  
  parameters: {
    candidate_labels: ['cat', 'dog']
  }
})
```

## Multimodal

### Feature Extraction

This task reads some text and outputs raw float values, that are usually consumed as part of a semantic database/semantic search.

```typescript
await hf.featureExtraction({
  model: "sentence-transformers/distilbert-base-nli-mean-tokens",
  inputs: "That is a happy person",
});
```

### Visual Question Answering

Visual Question Answering is the task of answering open-ended questions based on an image. They output natural language responses to natural language questions.

```typescript
await hf.visualQuestionAnswering({
  model: 'dandelin/vilt-b32-finetuned-vqa',
  inputs: {
    question: 'How many cats are lying down?',
    image: await (await fetch('https://placekitten.com/300/300')).blob()
  }
})
```

### Document Question Answering

Document question answering models take a (document, question) pair as input and return an answer in natural language.

```typescript
await hf.documentQuestionAnswering({
  model: 'impira/layoutlm-document-qa',
  inputs: {
    question: 'Invoice number?',
    image: await (await fetch('https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png')).blob(),
  }
})
```

## Tabular

### Tabular Regression

Tabular regression is the task of predicting a numerical value given a set of attributes.

```typescript
await hf.tabularRegression({
  model: "scikit-learn/Fish-Weight",
  inputs: {
    data: {
      "Height": ["11.52", "12.48", "12.3778"],
      "Length1": ["23.2", "24", "23.9"],
      "Length2": ["25.4", "26.3", "26.5"],
      "Length3": ["30", "31.2", "31.1"],
      "Species": ["Bream", "Bream", "Bream"],
      "Width": ["4.02", "4.3056", "4.6961"]
    },
  },
})
```

### Tabular Classification

Tabular classification is the task of classifying a target category (a group) based on set of attributes.

```typescript
await hf.tabularClassification({
  model: "vvmnnnkv/wine-quality",
  inputs: {
    data: {
      "fixed_acidity": ["7.4", "7.8", "10.3"],
      "volatile_acidity": ["0.7", "0.88", "0.32"],
      "citric_acid": ["0", "0", "0.45"],
      "residual_sugar": ["1.9", "2.6", "6.4"],
      "chlorides": ["0.076", "0.098", "0.073"],
      "free_sulfur_dioxide": ["11", "25", "5"],
      "total_sulfur_dioxide": ["34", "67", "13"],
      "density": ["0.9978", "0.9968", "0.9976"],
      "pH": ["3.51", "3.2", "3.23"],
      "sulphates": ["0.56", "0.68", "0.82"],
      "alcohol": ["9.4", "9.8", "12.6"]
    },
  },
})
```

You can use any Chat Completion API-compatible provider with the `chatCompletion` method.

```typescript
// Chat Completion Example
const MISTRAL_KEY = process.env.MISTRAL_KEY;
const hf = new HfInference(MISTRAL_KEY);
const ep = hf.endpoint("https://api.mistral.ai");
const stream = ep.chatCompletionStream({
  model: "mistral-tiny",
  messages: [{ role: "user", content: "Complete the equation one + one = , just the answer" }],
});
let out = "";
for await (const chunk of stream) {
  if (chunk.choices && chunk.choices.length > 0) {
    out += chunk.choices[0].delta.content;
    console.log(out);
  }
}
```


### Image Generation

You can get any images using pollinations.ai, Example

```js
// Image details
const prompt = 'A lone figure stands in the neon glow of a bustling metropol...';
const width = 512;
const height = 512;

const imageUrl = `https://pollinations.ai/p/${encodeURIComponent(prompt)}?width=${width}&height=${height}`;
```
