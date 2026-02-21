import * as tf from '@tensorflow/tfjs-node'

// ===============================
//1️⃣ Configurações do Modelo
// ===============================

const INPUT_FEATURES = 7
const EPOCHS = 200


// ===============================
// 2️⃣ Função para Criar o Modelo
// ===============================

function createModel(): tf.LayersModel {
  const model = tf.sequential()

  model.add(
    tf.layers.dense({
      inputShape: [INPUT_FEATURES],
      units: 16,
      activation: 'relu'
    })
  )

  model.add(
    tf.layers.dense({
      units: 8,
      activation: 'relu'
    })
  )

  model.add(
    tf.layers.dense({
      units: 1,
      activation: 'sigmoid' // binário
    })
  )

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  })

  return model
}


// ===============================
// 3️⃣ Função de Treinamento
// ===============================

async function trainModel(
  model: tf.LayersModel,
  inputXs: tf.Tensor,
  outputYs: tf.Tensor
) {
  await model.fit(inputXs, outputYs, {
    epochs: EPOCHS,
    shuffle: true,
    verbose: 1
  })
}


// ===============================
// 4️⃣ Função de Predição
// ===============================

async function predict(
  model: tf.LayersModel,
  clienteNormalizado: number[][]
) {
  const inputTensor = tf.tensor2d(clienteNormalizado)

  const prediction = model.predict(inputTensor) as tf.Tensor

  const probability = (await prediction.data())[0]

  return {
    probability,
    prediction: probability > 0.5 ? 'vai_cancelar' : 'vai_ficar'
  }
}

// ===============================
// 6️⃣ Normalização de Dados
// ===============================

type Plano = 'basic' | 'medium' | 'premium'

interface ClienteInput {
  tempo_cliente: number        // meses (ex: 12)
  logins_semana: number        // ex: 8
  tickets_suporte: number      // ex: 3
  plano: Plano                 // 'basic' | 'medium' | 'premium'
  atraso_pagamento: boolean    // true ou false
}

/**
 * Normalização Min-Max
 * Fórmula: (valor - min) / (max - min)
 */
function normalize(value: number, min: number, max: number): number {
  return (value - min) / (max - min)
}

/**
 * Converte plano para one-hot encoding
 */
function encodePlano(plano: Plano): number[] {
  switch (plano) {
    case 'basic':
      return [1, 0, 0]
    case 'medium':
      return [0, 1, 0]
    case 'premium':
      return [0, 0, 1]
  }
}


// Função principal de normalização

function normalizeCliente(cliente: ClienteInput): number[] {
  // Definição de limites (baseados em regra de negócio)
  const min_max_time = [0, 60] // meses de contrato
  const min_max_logins = [0, 50] // logins por semana
  const min_max_tickets = [0, 20] // tickets de suporte

  const tempoNormalizado = normalize(
    cliente.tempo_cliente,
    min_max_time[0],
    min_max_time[1]
  )

  const loginNormalizado = normalize(
    cliente.logins_semana,
    min_max_logins[0],
    min_max_logins[1]
  )

  const ticketsNormalizado = normalize(
    cliente.tickets_suporte,
    min_max_tickets[0],
    min_max_tickets[1]
  )

  const planoEncoded = encodePlano(cliente.plano)

  const atraso = cliente.atraso_pagamento ? 1 : 0

  return [
    tempoNormalizado,
    loginNormalizado,
    ticketsNormalizado,
    ...planoEncoded,
    atraso
  ]
}


// ===============================
// 5️⃣ Dataset de Treino
// ===============================
// Ordem das features:
// [
//  tempo_cliente,
//  logins_semana,
//  tickets_suporte,
//  plano_basic,
//  plano_medium,
//  plano_premium,
//  atraso_pagamento
// ]

function gerarClienteAleatorio(): ClienteInput {
  const planos: Plano[] = ['basic', 'medium', 'premium']

  return {
    tempo_cliente: Math.floor(Math.random() * 61), // 0–60 meses
    logins_semana: Math.floor(Math.random() * 51), // 0–50
    tickets_suporte: Math.floor(Math.random() * 21), // 0–20
    plano: planos[Math.floor(Math.random() * 3)],
    atraso_pagamento: Math.random() < 0.3 // 30% chance de atraso
  }
}


// Regra artificial de churn:
// - Pouco tempo de contrato (< 12 meses)
// - Poucos logins (< 10)
// - Muitos tickets (> 8)
// - Ou atraso_pagamento = true

function calcularChurnProbabilistico(cliente: ClienteInput): number {
  let z = 0

  z += -0.05 * cliente.tempo_cliente
  z += -0.08 * cliente.logins_semana
  z += 0.15 * cliente.tickets_suporte
  z += cliente.atraso_pagamento ? 1.2 : 0

  const prob = 1 / (1 + Math.exp(-z))

  return Math.random() < prob ? 1 : 0
}

function gerarDataset(quantidade: number) {
  const inputs: number[][] = []
  const labels: number[][] = []

  for (let i = 0; i < quantidade; i++) {
    const cliente = gerarClienteAleatorio()
    const clienteNormalizado = normalizeCliente(cliente)
    const label = calcularChurnProbabilistico(cliente)

    inputs.push(clienteNormalizado)
    labels.push([label])
  }

  return { inputs, labels }
}

async function main() {
  const { inputs, labels } = gerarDataset(1000)

  const inputXs = tf.tensor2d(inputs)
  const outputYs = tf.tensor2d(labels)

  const model = createModel()

  await trainModel(model, inputXs, outputYs)
  
  // Novo cliente para prever
  const novoClienteRaw: ClienteInput = {
    tempo_cliente: 25,        // meses (0 a 60 meses) de contrato
    logins_semana: 10,         // 0 a 50 logins
    tickets_suporte: 18,       // 0 a 20 tickets
    plano: 'premium',          // "basic" | "medium" | "premium"
    atraso_pagamento: true   // true / false
  }

  const novoClienteNormalizado = [
    normalizeCliente(novoClienteRaw)
  ]

  const result = await predict(model, novoClienteNormalizado)

  console.log('\nResultado da previsão:')
  console.log(`Probabilidade de cancelar: ${(result.probability * 100).toFixed(2)}%`)
  console.log(`Classificação: ${result.prediction}`)
}

main()