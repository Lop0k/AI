const tf = require('@tensorflow/tfjs-node');
const { data, targets } = tf.tidy(() => {
  const mnist = require('mnist-data');
  const data = mnist.training(60000);
  const targets = data.labels;
  const { images } = data;
  return {
    data: images.reshape([60000, 28, 28, 1]),
    targets: targets
  };
});

const model = tf.sequential();

model.add(tf.layers.flatten({ inputShape: [28, 28, 1] }));
model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
model.add(tf.layers.dropout({ rate: 0.2 }));
model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

model.compile({
  optimizer: 'adam',
  loss: 'sparseCategoricalCrossentropy',
  metrics: ['accuracy']
});

model.fit(data, targets, { epochs: 5 });

const test_data = require('mnist-data').testing(10000);
const test_targets = test_data.labels;
const test_images = test_data.images.reshape([10000, 28, 28, 1]);

const evalOutput = model.evaluate(test_images, test_targets);
console.log('Evaluation result:', evalOutput[1]);
