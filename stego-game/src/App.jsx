import { useState } from 'react'
import './App.css'

function App() {
  const [prompt, setPrompt] = useState('')
  const [messages, setMessages] = useState(null)
  const [correctAnswer, setCorrectAnswer] = useState(null)
  const [userAnswer, setUserAnswer] = useState(null)
  const [showResult, setShowResult] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)

  // Mock function to generate text (will be replaced with actual API call later)
  const generateMockText = (promptText) => {
    const responses = [
      "Mitochondria are the cell's power plants — tiny, double-membrane organelles that turn fuel into usable energy. They take in nutrients like glucose and fatty acids, break them down through a process called cellular respiration, and produce ATP (adenosine triphosphate), which is the chemical energy that powers nearly every process in the cell.",
      "Mitochondria are small structures inside most of your cells that act as the cell's \"power plants.\" They convert the energy from food (mainly from sugars and fats) into a form of chemical energy called ATP (adenosine triphosphate) that your cells can actually use to function. Here are a few key things about them...",
      "The mitochondrion is a double-membrane-bound organelle found in most eukaryotic cells. It generates most of the cell's supply of adenosine triphosphate (ATP), which is used as a source of chemical energy. Mitochondria contain their own DNA and ribosomes, which suggests they originated from ancient bacterial cells.",
      "Think of mitochondria as tiny power generators inside each cell. They work 24/7 to convert nutrients from the food you eat into ATP, the energy currency your body uses for everything from thinking to running. Without mitochondria, complex life as we know it wouldn't exist.",
      "Mitochondria perform cellular respiration to create ATP through a series of chemical reactions. The process involves glycolysis, the citric acid cycle, and oxidative phosphorylation. These organelles are essential for life and are involved in many cellular processes beyond energy production, including signaling, cellular differentiation, and cell death."
    ]

    return responses[Math.floor(Math.random() * responses.length)]
  }

  // Mock function to encode a hidden message using steganography
  const encodeWithSteganography = (text, hiddenMessage) => {
    // For demo purposes, we'll just add a subtle marker
    // In reality, this would use actual steganography techniques
    return text + `<!-- ${hiddenMessage} -->`
  }

  const handleGenerate = () => {
    if (!prompt.trim()) return

    setIsGenerating(true)
    setShowResult(false)
    setUserAnswer(null)

    // Simulate API delay
    setTimeout(() => {
      // Generate two messages
      const vanillaMessage = generateMockText(prompt)
      const stegoMessage = generateMockText(prompt)
      const hiddenMessage = prompt // The hidden message is the original prompt

      // Randomly decide which position (A or B) gets the stego message
      const stegoPosition = Math.random() > 0.5 ? 'A' : 'B'

      setMessages({
        A: stegoPosition === 'A' ? encodeWithSteganography(stegoMessage, hiddenMessage) : vanillaMessage,
        B: stegoPosition === 'B' ? encodeWithSteganography(stegoMessage, hiddenMessage) : vanillaMessage,
      })

      setCorrectAnswer(stegoPosition)
      setIsGenerating(false)
    }, 1000)
  }

  const handleAnswer = (answer) => {
    setUserAnswer(answer)
    setShowResult(true)
  }

  const handleReset = () => {
    setPrompt('')
    setMessages(null)
    setCorrectAnswer(null)
    setUserAnswer(null)
    setShowResult(false)
  }

  const isCorrect = userAnswer === correctAnswer

  return (
    <div className="app">
      <div className="container">
        <div className="input-section">
          <input
            type="text"
            className="prompt-input"
            placeholder="what would you like to say?"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleGenerate()}
            disabled={isGenerating || messages !== null}
          />
          {!messages && (
            <button
              className="generate-btn"
              onClick={handleGenerate}
              disabled={!prompt.trim() || isGenerating}
            >
              {isGenerating ? 'Generating...' : 'Generate'}
            </button>
          )}
        </div>

        {messages && (
          <>
            <div className="messages-section">
              <div className="message-box">
                <div className="message-content">
                  {messages.A.replace(/<!--.*-->/, '')}
                </div>
              </div>

              <div className="message-box">
                <div className="message-content">
                  {messages.B.replace(/<!--.*-->/, '')}
                </div>
              </div>
            </div>

            <div className="question">
              Which one has the hidden message?
            </div>

            {!showResult && (
              <div className="answer-buttons">
                <button
                  className="answer-btn answer-a"
                  onClick={() => handleAnswer('A')}
                >
                  A
                </button>
                <button
                  className="answer-btn answer-b"
                  onClick={() => handleAnswer('B')}
                >
                  B
                </button>
              </div>
            )}

            {showResult && (
              <div className="result-section">
                <div className={`result ${isCorrect ? 'correct' : 'wrong'}`}>
                  {isCorrect ? 'Correct!' : 'Wrong!'}
                </div>
                <div className="hidden-message">
                  Hidden message: "{prompt}"
                  <br />
                  The steganography was in option {correctAnswer}
                </div>
                <button className="reset-btn" onClick={handleReset}>
                  Try Again
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default App
