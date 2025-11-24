import { useState } from 'react'
import './App.css'

function App() {
  const [prompt, setPrompt] = useState('')
  const [hiddenMessage, setHiddenMessage] = useState('attack at dawn')
  const [messages, setMessages] = useState(null)
  const [correctAnswer, setCorrectAnswer] = useState(null)
  const [userAnswer, setUserAnswer] = useState(null)
  const [showResult, setShowResult] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)

  const handleGenerate = async () => {
    if (!prompt.trim() || !hiddenMessage.trim()) return

    setIsGenerating(true)
    setShowResult(false)
    setUserAnswer(null)

    try {
      // Call the backend API (use ngrok URL for public access)
      const backendUrl = 'https://84ab38e0843a.ngrok-free.app/api/generate';
      const response = await fetch(backendUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: prompt,
          hidden_message: hiddenMessage,
          temperature: 1.3,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      console.table(data)

      // Randomly decide which position (A or B) gets the stego message
      const stegoPosition = Math.random() > 0.5 ? 'A' : 'B'

      setMessages({
        A: stegoPosition === 'A' ? data.stego_message : data.vanilla_message,
        B: stegoPosition === 'B' ? data.stego_message : data.vanilla_message,
      })

      setCorrectAnswer(stegoPosition)
      setIsGenerating(false)
    } catch (error) {
      console.error('Error generating messages:', error)
      alert('Error generating messages. Please make sure the backend server is running.')
      setIsGenerating(false)
    }
  }

  const handleAnswer = (answer) => {
    setUserAnswer(answer)
    setShowResult(true)
  }

  const handleReset = () => {
    setPrompt('')
    setHiddenMessage('attack at dawn')
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
          <label className="input-label">Prompt</label>
          <input
            type="text"
            className="prompt-input"
            placeholder="Enter prompt (e.g., 'tell me a story')"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleGenerate()}
            disabled={isGenerating || messages !== null}
          />
          <label className="input-label">Hidden Message</label>
          <input
            type="text"
            className="prompt-input"
            placeholder="Enter hidden message"
            value={hiddenMessage}
            onChange={(e) => setHiddenMessage(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleGenerate()}
            disabled={isGenerating || messages !== null}
          />
          {!messages && !isGenerating && (
            <button
              className="generate-btn"
              onClick={handleGenerate}
              disabled={!prompt.trim() || !hiddenMessage.trim()}
            >
              Generate
            </button>
          )}
          {isGenerating && (
            <div className="loading-message">
              Generating, usually takes 30 seconds...
            </div>
          )}
        </div>

        {messages && (
          <>
            <div className="messages-section">
              <div className="message-box">
                <div className="message-content">
                  {messages.A}
                </div>
              </div>

              <div className="message-box">
                <div className="message-content">
                  {messages.B}
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
                  Hidden message: "{hiddenMessage}"
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
