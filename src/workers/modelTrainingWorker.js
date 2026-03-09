import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');

let _MODEL = null;
let _PRODUTCS_CONTEXT = null;
let _PREPROCESSED_DATA = null;

// 🔢 Normalize continuous values (price, age) to 0–1 range
// Why? Keeps all features balanced so no one dominates training
// Formula: (val - min) / (max - min)
// Example: price=129.99, minPrice=39.99, maxPrice=199.99 → 0.56
const normalize = (value, min, max) => (value - min) / ((max - min) || 1)

const WEIGHTS = {
    category: 0.7,
    price: 0.5,
    color: 0.3,
    age: 0.2
};

// Sistema de recomendação de produtos baseado em aprendizado de máquina
// recomendar produtos com base nos precos, cores, categorias, idades etc
// Ex: Se usuarios da faixa de 25-30 compram mais um produto, ele vai ser recomenado para outros usuarios dessa faixa etaria
// Como a ideia é recomendar produtos para um usuário, então, a rede precisa aprender com os usuários já existem
// Assim, a entrada da rede deve ser um usuario e os produtos que ele comprou. Com isso ela aprende o padrão dos produtos que tal perfil de usuario compra
// Com base nisso, é possível recomendar produtos semalhantes caso o usário tiver o mesmo perfil

// Logo, precisamos normalizar os dados dos usuarios e suas compras. Isso vai formar uma entrada.
// O gabarito da rede neural vai ser o próprio produto que usuário comprou. 
// Dessa forma, teremos varios inputs, em que cada um tem uma combinação de usuário e um produto. 
// O usuário vai ter input para cada produto. Se o produto for comprado pelo usuário, então o label é 1.
// Caso contrário o label é zero. Assim se repete para todos os usuários.
// O resultado final é nossos dados de treinamento.

function preprocessData(users, products) {
    const ages = users.map(u => u.age);
    const prices = products.map(p => p.price);

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    const colors = [...new Set(products.map(p => p.color))];
    const categories = [...new Set(products.map(p => p.category))];

    const colorIndex = Object.fromEntries(
        colors.map((color, index) => [color, index])
    )

    const categoryIndex = Object.fromEntries(
        categories.map((cat, index) => [cat, index])
    )

    // calcular a media de idade para um produto
    const sumAge = {}
    const countAge = {}
    users.forEach(user => {
        user.purchases.forEach(p => {
            sumAge[p.name] = (sumAge[p.name] || 0) + user.age;
            countAge[p.name] = (countAge[p.name] || 0) + 1;
        })
    });

    // gerar um objeto com chave e valor
    // chave é o nome do produto e o valor media de idade do produto
    const midAge = ages.reduce((total, actual) => total + actual, 0) / ages.length;
    const productAvgAge = Object.fromEntries(
        products.map(p => {
            const avgAge = sumAge[p.name] ? sumAge[p.name] / countAge[p.name] : midAge;

            return [p.name, avgAge]
        }
        ));

    const categoryLength = Object.keys(categoryIndex).length;
    const colorLength = Object.keys(colorIndex).length;

    return {
        ages,
        prices,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        colorIndex,
        colorLength: colorLength,
        categoryIndex,
        categoryLength: categoryLength,
        productAvgAge,
        products,
        users,
        dimension: 1 + 1 + categoryLength + colorLength // price, age, category and color
    };
}


const oneHotWeighted = (index, length, weight) =>
    tf.oneHot(index, length).cast('float32').mul(weight)

function normalizeProduct(product, preprocessedData) {
    const priceNormalized = normalize(product.price, preprocessedData.minPrice, preprocessedData.maxPrice) * WEIGHTS.price
    const price = tf.tensor1d([priceNormalized])

    const ageNormalized = preprocessedData.productAvgAge[product.name] * WEIGHTS.age;
    const age = tf.tensor1d([ageNormalized])

    const category = oneHotWeighted(
        preprocessedData.categoryIndex[product.category],
        preprocessedData.categoryLength,
        WEIGHTS.category
    )

    const color = oneHotWeighted(
        preprocessedData.colorIndex[product.color],
        preprocessedData.colorLength,
        WEIGHTS.color
    )

    return tf.concat1d(
        [price, age, category, color]
    )
}

// codificar o usuario para a rede neural
// objetivo: o usuário é definido pelo o que ele compra
// assim, vamos codificar o usuario conforme suas compras
// se ja tiver compras, definimos ele pelos produtos que ele comprou
// se é um novo usuário ou usuario que nunca comprou, definimos tudo zerado sem perfil nenhum

// Redes neurais operam com o conceito de lotes (batches). 
// Elas não esperam receber um vetor solto de um único usuário; elas esperam uma matriz onde cada linha é um usuário.
// Por isso, o uso do reshape.
function normalizeUser(user, preprocessedData) {
    if (user.purchases.length == 0) {
        return tf.concat1d(
            [
                tf.zeros([1]), // preço é zerado
                tf.tensor1d([
                    normalize(user.age, preprocessedData.minAge, preprocessedData.maxAge)
                ]),
                tf.zeros([preprocessedData.categoryLength]),
                tf.zeros([preprocessedData.colorLength])
            ]
        );
    }

    // O .mean(0) diz ao TensorFlow: "Calcule a média esmagando a dimensão 0 (as linhas)".
    return tf.stack(
        user.purchases.map(product => normalizeProduct(product, preprocessedData))
    ).mean(0);
}

// normaliza os dados para serem usados na rede neural
// a ideia é montar o input e o output
// o output que queremos é produto que vou recomendar para o usuario
// então, temos um novo usuário e preciso mostrar uma lista de produtos que fazem sentido para aquele perfil de usuário
// Assim, a rede precisa gerar como output o produto ideal para aquele usuario
// Portanto, precisamos gerar combinações entre usuarios e produtos
// Vamos fazer uma combinação de cada usuario com todos os produtos
// Logo, um usuario vai ter relação com todos os produtos e a saida vai ser 1 ou 0
// 1 se ele comprou o produto e 0 senao comprou
// Dessa forma, a rede neural sabe pelo perfil do usuario se o produto é mais comprado ou não
// O resultado do modelo consegue gerar essa relação entre usuário e produto

// Isso surge um problema, pois se tiver muitos produtos no sistema, o processamento vai ficar muito lento
// E também se tiver muitos produtos, a maioria das saidas vao ser 0 e a IA começa a recomendar nenhum produto
// Assim, vamos usar o Negative Sampling, ou seja, limitaremos a quantidade de produtos que ele não comprou
// A relação deve manter 1:3 (arbitrario). Se ele comprou 2 produtos, vamos sortear aleatoriamente 6 produtos que não comprou.
// Limitamos a função para O(U * K), tornando muito mais performática

function drawNegatives(products, userPurchases, K) {
    const namesPurchased = new Set(userPurchases.map(p => p.name));

    const negativeDrawn = [];
    const indicesDrawn = new Set();

    // evitar looping infinito para usuarios que comprou todos os produtos
    const maxAttempts = products.length;
    let attempts = 0;

    while (negativeDrawn.length < K && attempts < maxAttempts) {
        const randomIndex = Math.floor(Math.random() * products.length);
        const candidateProduct = products[randomIndex];

        const purchased = namesPurchased.has(candidateProduct.name);
        const drawn = indicesDrawn.has(randomIndex);

        if (!purchased && !drawn) {
            negativeDrawn.push(candidateProduct);
            indicesDrawn.add(randomIndex);
        }

        attempts++;
    }

    return negativeDrawn;
}

function buildTrainingData(preprocessedData) {
    const inputs = []
    const labels = []

    const K = 200;

    const users = preprocessedData.users;
    const products = preprocessedData.products;

    users.filter(u => u.purchases.length)
        .forEach(user => {
            const userTensor = normalizeUser(user, preprocessedData);

            const productsPurchased = user.purchases;
            const productsNotPurchased = drawNegatives(products, user.purchases, K);

            productsPurchased.forEach(product => {
                const productTensor = normalizeProduct(product, preprocessedData);
                inputs.push(tf.concat1d([userTensor, productTensor]))
                labels.push(1);
            })

            productsNotPurchased.forEach(product => {
                const productTensor = normalizeProduct(product, preprocessedData);
                inputs.push(tf.concat1d([userTensor, productTensor]))
                labels.push(0);
            })
        });

    return {
        xs: tf.stack(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimension: preprocessedData.dimension * 2 // usuario + produto
    }
}

// recebe dados de entrada (xs) e saida (ys) e tamanho do input (input dimension)
async function configureModelAndTrain(trainingData) {
    const model = tf.sequential();

    // camda de entrada
    // camada densa, todos os neuronios conectam com as saidas da camada anterior
    model.add(
        tf.layers.dense({
            inputShape: [trainingData.inputDimension],
            units: 128,
            activation: 'relu'
        })
    )

    // camada oculta
    model.add(
        tf.layers.dense({
            units: 32,
            activation: 'relu'
        })
    )

     // camada oculta
    model.add(
        tf.layers.dense({
            units: 8,
            activation: 'relu'
        })
    )

    // camada de saída
    model.add(
        tf.layers.dense({ units: 1, activation: 'sigmoid' })
    )

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    })

    console.log("Iniciando o treinamento na fábrica...");

    await model.fit(trainingData.xs, trainingData.ys, {
        epochs: 50,           // Quantas vezes o aluno vai ler o caderno
        batchSize: 32,        // Quantos exercícios ele lê antes de ajustar o cérebro
        validationSplit: 0.2, // A Mágica: 20% dos dados vão para o cofre (A Prova)
        shuffle: true,        // Embaralha os dados de treino para evitar decoreba

        // O "Boletim Escolar" em tempo real
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Rodada ${epoch + 1}:`);
                console.log(`  Nota no Treino (Estudo): ${(logs.acc * 100).toFixed(1)}%`);
                console.log(`  Nota na Validação (Prova): ${(logs.val_acc * 100).toFixed(1)}%`);
                console.log(`  Loss (Erro): ${(logs.loss * 100).toFixed(1)}%`);

                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch,
                    loss: logs.loss,
                    accuracy: logs.val_acc
                });
            }
        }
    });

    console.log("Treinamento concluído!");

    return model;
}

async function trainModel({ users }) {
    console.log('Training model with users:', users);
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 1 } });

    var products = await (await fetch('/data/products.json')).json();
    console.log('Products loaded for training:', products);

    const preprocessedData = preprocessData(users, products);

    _PREPROCESSED_DATA = preprocessedData;

    // Em aplicações reais:
    //  Armazene todos os vetores de produtos em um banco de dados vetorial (como Postgres, Neo4j ou Pinecone)
    _PRODUTCS_CONTEXT = products.map(product => {
        return {
            name: product.name,
            meta: { ...product },
            tensor: normalizeProduct(product, preprocessedData)
        }
    });

    console.log(preprocessedData)

    const trainingData = buildTrainingData(preprocessedData);

    _MODEL = await configureModelAndTrain(trainingData);

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}

function recommend({ user }) {
    if (!_MODEL) return;

    if (!user) {
        console.log("user nao definido")
        return;
    }

    if (!_PREPROCESSED_DATA) {
        console.log("dados pre processados do modelo não foi gerado")
        return;
    }

    console.log('will recommend for user:', user)

    const userTensor = normalizeUser(user, _PREPROCESSED_DATA);

    // Em aplicações reais:
    //  Consulta base vetorizada: Encontre os 200 produtos mais próximos do vetor do usuário
    //  Execute _model.predict() apenas nesses produtos

    const inputs = _PRODUTCS_CONTEXT.map(productCtx =>
        tf.concat1d([userTensor, productCtx.tensor])
    )

    const xsInput = tf.stack(inputs);
    const predictions = _MODEL.predict(xsInput)
    const scores = predictions.dataSync();

    const recommendations = _PRODUTCS_CONTEXT.map((item, index) => {
        return {
            ...item.meta,
            name: item.name,
            score: scores[index]
        }
    })

    const sortedItems = recommendations.sort((a, b) => b.score - a.score);

    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedItems
    });
}

const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: user => recommend(user),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
