# Backend Integration Guide - Steganography Guessing Game

## Overview

This document explains the frontend implementation of the steganography guessing game and provides guidance for backend integration.

## Game Concept

The game presents users with two AI-generated text responses (A and B) based on a user prompt. One response contains a hidden message encoded using steganography, while the other is a standard AI response. The user must guess which one contains the hidden message.

## Frontend Architecture

### Technology Stack
- **React 19.2.0** with Hooks
- **Vite 7.2.2** as build tool
- **pnpm** for package management

### State Management

The application uses React's `useState` hooks to manage the following state:

```javascript
const [prompt, setPrompt] = useState('')           // User's input prompt
const [messages, setMessages] = useState(null)     // Object containing A and B messages
const [correctAnswer, setCorrectAnswer] = useState(null)  // 'A' or 'B' - which has stego
const [userAnswer, setUserAnswer] = useState(null) // User's guess ('A' or 'B')
const [showResult, setShowResult] = useState(false) // Whether to show correct/wrong
const [isGenerating, setIsGenerating] = useState(false) // Loading state
```

### Game Flow

1. **Input Phase**: User enters a prompt in the text input
2. **Generation Phase**: Click "Generate" button triggers message generation
3. **Display Phase**: Two messages (A and B) are displayed side-by-side
4. **Guess Phase**: User clicks either A or B button to guess
5. **Result Phase**: App reveals if correct/wrong and shows the hidden message
6. **Reset**: User can click "Try Again" to start over

## Backend API Requirements

### Endpoint 1: Generate Messages

The backend should provide an endpoint to generate both messages simultaneously.

**Suggested Endpoint**: `POST /api/generate`

**Request Body**:
```json
{
  "prompt": "string",        // The user's input prompt (also serves as hidden message)
  "temperature": 0.7,        // Or whatever default you choose
  "model": "your-model-name" // Optional: specify which AI model to use
}
```

**Response Body**:
```json
{
  "vanilla_message": "string",  // Standard AI-generated response
  "stego_message": "string",    // Response with hidden message encoded
  "hidden_message": "string"    // The message that was hidden (for verification)
}
```

### Important Considerations

#### 1. **Consistent Generation Parameters**
Both the vanilla and steganography messages MUST use identical generation parameters:
- Same temperature
- Same model
- Same system prompts (if any)
- Same max_tokens/length limits

This ensures the game is fair and the differences are only due to steganography, not generation settings.

#### 2. **Randomization on Frontend**
The frontend randomly assigns which message (vanilla vs stego) goes to position A or B. The backend doesn't need to handle this - just return both versions.

#### 3. **Hidden Message**
Currently, the frontend uses the user's original prompt as the hidden message. You may want to:
- Keep this approach (simple, makes sense)
- Allow customization (let users specify a different hidden message)
- Generate a random hidden message

## Current Mock Implementation

### Location: `src/App.jsx`

#### Mock Text Generation (Lines 13-23)
```javascript
const generateMockText = (promptText) => {
  const responses = [
    "Mitochondria are the cell's power plants...",
    // ... 5 pre-written responses
  ]
  return responses[Math.floor(Math.random() * responses.length)]
}
```

**To Replace**: This entire function should be replaced with an actual API call to your backend.

#### Mock Steganography (Lines 26-30)
```javascript
const encodeWithSteganography = (text, hiddenMessage) => {
  // For demo purposes, we'll just add a subtle marker
  return text + `<!-- ${hiddenMessage} -->`
}
```

**To Replace**: This is just a placeholder. Your backend should handle the actual steganography encoding.

#### Generation Handler (Lines 32-57)
```javascript
const handleGenerate = () => {
  if (!prompt.trim()) return

  setIsGenerating(true)
  setShowResult(false)
  setUserAnswer(null)

  // Simulate API delay
  setTimeout(() => {
    const vanillaMessage = generateMockText(prompt)
    const stegoMessage = generateMockText(prompt)
    const hiddenMessage = prompt

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
```

## Integration Instructions

### Step 1: Replace Mock Generation with API Call

Replace the `handleGenerate` function's setTimeout block with an actual API call:

```javascript
const handleGenerate = async () => {
  if (!prompt.trim()) return

  setIsGenerating(true)
  setShowResult(false)
  setUserAnswer(null)

  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt: prompt,
        temperature: 0.7, // Adjust as needed
      }),
    })

    const data = await response.json()

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
    setIsGenerating(false)
    // TODO: Add error handling UI
  }
}
```

### Step 2: Remove Mock Functions

Once integrated, you can safely remove:
- `generateMockText()` function (lines 13-23)
- `encodeWithSteganography()` function (lines 26-30)

### Step 3: Add Error Handling

Consider adding user-facing error messages:
```javascript
const [error, setError] = useState(null)

// In handleGenerate catch block:
setError('Failed to generate messages. Please try again.')

// In JSX:
{error && <div className="error-message">{error}</div>}
```

### Step 4: Environment Configuration

Add API endpoint configuration:
```javascript
// Create a config file or use environment variables
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3000'
```

## UI Components

### Input Section
- Single text input with placeholder: "what would you like to say?"
- Generate button (disabled when empty or generating)
- Shows "Generating..." text during API call

### Messages Section
- Two side-by-side boxes (A and B)
- Rounded corners, bordered design
- Displays the full AI-generated text
- No indication of which has steganography

### Answer Section
- Question: "Which one has the hidden message?"
- Two buttons: A (red) and B (green)
- Only shown before user makes a selection

### Result Section
- Shows "Correct!" (green) or "Wrong!" (red)
- Reveals the hidden message
- Shows which option (A or B) had the steganography
- "Try Again" button to reset the game

## Styling Notes

- Font: Comic Sans MS (playful, casual feel)
- Color scheme:
  - Red (#dc3545) for option A
  - Green (#28a745) for option B
  - Blue (#007bff) for reset button
- Responsive design with mobile breakpoint at 768px
- Smooth transitions and animations

## Testing Checklist

### Backend Testing
- [ ] API returns both vanilla and stego messages
- [ ] Both messages use identical generation parameters
- [ ] Steganography encoding is working correctly
- [ ] API handles errors gracefully
- [ ] Response times are reasonable (< 5 seconds)

### Integration Testing
- [ ] Frontend successfully calls backend API
- [ ] Messages display correctly in UI
- [ ] Randomization works (A/B assignment is unpredictable)
- [ ] Correct answer detection works
- [ ] Error states are handled
- [ ] Loading states display properly

### End-to-End Testing
- [ ] User can enter prompt and generate messages
- [ ] User can make a guess
- [ ] Result shows correct/wrong accurately
- [ ] Hidden message is revealed correctly
- [ ] Reset functionality works
- [ ] Game can be played multiple times in succession

## Future Enhancements

### Frontend
- Add difficulty levels (easier/harder steganography)
- Track score across multiple rounds
- Add a timer for competitive play
- Show hints or detection tips
- Add user accounts and leaderboards

### Backend
- Multiple steganography techniques to choose from
- Adjustable steganography strength
- Custom hidden message input (separate from prompt)
- Analytics on detection rates
- A/B testing different AI models

## File Structure

```
stego-game/
├── src/
│   ├── App.jsx         # Main game component (NEEDS API INTEGRATION)
│   ├── App.css         # Game-specific styles
│   ├── index.css       # Global styles
│   └── main.jsx        # React entry point
├── public/             # Static assets
├── package.json        # Dependencies
└── vite.config.js      # Vite configuration
```

## Questions for Backend Team

1. **What steganography technique will you use?**
   - Least Significant Bit (LSB)?
   - Synonym substitution?
   - Whitespace encoding?
   - Custom algorithm?

2. **What AI model will power the text generation?**
   - OpenAI GPT?
   - Anthropic Claude?
   - Open-source model?

3. **Expected response time?**
   - Should we add progress indicators?
   - Should we implement streaming responses?

4. **Rate limiting?**
   - How many generations per user/session?
   - Should we implement client-side throttling?

5. **Authentication?**
   - Will this require user authentication?
   - Should we implement session management?

## Contact

For questions or clarifications about the frontend implementation, please reach out to the frontend team.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-12
**Status**: Ready for backend integration
