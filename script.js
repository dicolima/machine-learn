const STATUS = document.getElementById('status');
const VIDEO = document.getElementById('webcam');
const ENABLE_CAM_BUTTON = document.getElementById('enableCam');
const RESET_BUTTON = document.getElementById('reset');
const TRAIN_BUTTON = document.getElementById('train');
const SET_CLASSES_BUTTON = document.getElementById('setClasses');
const GENERATE_INPUTS_BUTTON = document.getElementById('generateInputs');
const CLASS_INPUTS_CONTAINER = document.getElementById('classInputsContainer');
const BUTTONS_CONTAINER = document.getElementById('buttonsContainer');
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;

let CLASS_NAMES = [];
let dataCollectorButtons = [];
let gatherDataState = STOP_DATA_GATHER;
let videoPlaying = false;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let predict = false;
let mobilenet;
let model;

// Gera os campos para os nomes das classes
GENERATE_INPUTS_BUTTON.addEventListener('click', () => {
  CLASS_INPUTS_CONTAINER.innerHTML = '';
  BUTTONS_CONTAINER.innerHTML = '';
  CLASS_NAMES = [];
  dataCollectorButtons = [];

  const numClasses = parseInt(document.getElementById('numClasses').value);
  if (numClasses < 2 || numClasses > 10) {
    alert('Escolha entre 2 e 10 classes.');
    return;
  }

  for (let i = 0; i < numClasses; i++) {
    const div = document.createElement('div');
    div.classList.add('class-input');
    div.innerHTML = `
      <label>Nome da Classe ${i + 1}: </label>
      <input type="text" id="className${i}" placeholder="Ex: Objeto ${i + 1}">
    `;
    CLASS_INPUTS_CONTAINER.appendChild(div);
  }

  SET_CLASSES_BUTTON.disabled = false;
});

// Define os nomes das classes e cria botões de coleta
SET_CLASSES_BUTTON.addEventListener('click', () => {
  CLASS_NAMES = [];
  BUTTONS_CONTAINER.innerHTML = '';
  dataCollectorButtons = [];

  const inputs = document.querySelectorAll('#classInputsContainer input');
  for (let i = 0; i < inputs.length; i++) {
    const name = inputs[i].value.trim() || `Classe ${i + 1}`;
    CLASS_NAMES.push(name);
  }

  // Criar botões de coleta
  CLASS_NAMES.forEach((name, index) => {
    const btn = document.createElement('button');
    btn.innerText = `Coletar dados de ${name}`;
    btn.classList.add('dataCollector');
    btn.dataset.onehot = index;
    btn.addEventListener('mousedown', gatherDataForClass);
    btn.addEventListener('mouseup', gatherDataForClass);
    BUTTONS_CONTAINER.appendChild(btn);
    dataCollectorButtons.push(btn);
  });

  STATUS.innerText = `Classes definidas: ${CLASS_NAMES.join(', ')}. Agora colete os dados!`;
});

// Ativa a webcam
ENABLE_CAM_BUTTON.addEventListener('click', enableCam);
TRAIN_BUTTON.addEventListener('click', trainAndPredict);
RESET_BUTTON.addEventListener('click', reset);

function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function enableCam() {
  if (hasGetUserMedia()) {
    const constraints = { video: true, width: 640, height: 480 };
    navigator.mediaDevices.getUserMedia(constraints).then(stream => {
      VIDEO.srcObject = stream;
      VIDEO.addEventListener('loadeddata', () => {
        videoPlaying = true;
        ENABLE_CAM_BUTTON.classList.add('removed');
      });
    });
  } else {
    console.warn('getUserMedia() não suportado neste navegador.');
  }
}

// Carrega o MobileNet
async function loadMobileNetFeatureModel() {
  const URL = 'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
  mobilenet = await tf.loadGraphModel(URL, { fromTFHub: true });
  STATUS.innerText = 'MobileNet v3 carregado com sucesso!';

  tf.tidy(() => {
    mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
  });
}
loadMobileNetFeatureModel();

// Loop de coleta de dados
function gatherDataForClass() {
  const classNumber = parseInt(this.dataset.onehot);
  gatherDataState = (gatherDataState === STOP_DATA_GATHER) ? classNumber : STOP_DATA_GATHER;
  dataGatherLoop();
}

function dataGatherLoop() {
  if (videoPlaying && gatherDataState !== STOP_DATA_GATHER) {
    let imageFeatures = tf.tidy(() => {
      let frame = tf.browser.fromPixels(VIDEO);
      let resized = tf.image.resizeBilinear(frame, [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH], true);
      return mobilenet.predict(resized.div(255).expandDims()).squeeze();
    });

    trainingDataInputs.push(imageFeatures);
    trainingDataOutputs.push(gatherDataState);

    if (!examplesCount[gatherDataState]) examplesCount[gatherDataState] = 0;
    examplesCount[gatherDataState]++;

    STATUS.innerText = CLASS_NAMES.map((name, i) =>
      `${name}: ${examplesCount[i] || 0} imagens`
    ).join(' | ');

    window.requestAnimationFrame(dataGatherLoop);
  }
}

// Treina e faz predições
async function trainAndPredict() {
  if (CLASS_NAMES.length < 2) {
    alert('Defina pelo menos duas classes antes de treinar.');
    return;
  }

  model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [1024], units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: CLASS_NAMES.length, activation: 'softmax' }));
  model.compile({
    optimizer: 'adam',
    loss: (CLASS_NAMES.length === 2) ? 'binaryCrossentropy' : 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
  const outputsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
  const oneHotOutputs = tf.oneHot(outputsTensor, CLASS_NAMES.length);
  const inputsTensor = tf.stack(trainingDataInputs);

  await model.fit(inputsTensor, oneHotOutputs, {
    shuffle: true,
    batchSize: 5,
    epochs: 10,
    callbacks: { onEpochEnd: (epoch, logs) => console.log(`Época ${epoch}:`, logs) }
  });

  outputsTensor.dispose();
  oneHotOutputs.dispose();
  inputsTensor.dispose();

  STATUS.innerText = 'Treinamento concluído! Fazendo predições...';
  predict = true;
  predictLoop();
}

// Loop de predição
function predictLoop() {
  if (predict) {
    tf.tidy(() => {
      let frame = tf.browser.fromPixels(VIDEO).div(255);
      let resized = tf.image.resizeBilinear(frame, [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH], true);
      let features = mobilenet.predict(resized.expandDims());
      let prediction = model.predict(features).squeeze();
      let index = prediction.argMax().arraySync();
      let probs = prediction.arraySync();

      STATUS.innerText = `Predição: ${CLASS_NAMES[index]} (${(probs[index] * 100).toFixed(1)}% de confiança)`;
    });
    window.requestAnimationFrame(predictLoop);
  }
}

// Resetar
function reset() {
  predict = false;
  examplesCount = [];
  trainingDataInputs.forEach(t => t.dispose());
  trainingDataInputs = [];
  trainingDataOutputs = [];
  STATUS.innerText = 'Dados limpos. Pronto para novo treinamento.';
  console.log('Tensores na memória:', tf.memory().numTensors);
}
