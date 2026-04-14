'use client'
import { useState, useEffect } from 'react'

const FIELD_LABELS = {
  invoice_number: 'Invoice number',
  invoice_date:   'Invoice date',
  due_date:       'Due date',
  issuer:         'Issuer',
  recipient:      'Recipient',
  total:          'Total amount'
}

export default function Home() {
  const [step, setStep] = useState('upload')
  const [result, setResult] = useState(null)

  useEffect(() => {
    if (step === 'processing') {
      setTimeout(() => {
        setResult({
          category: 'invoice',
          confidence: 97,
          fields: {
            invoice_number: 'INV-2024-0342',
            invoice_date: '2024-03-15',
            due_date: '2024-04-15',
            issuer: 'TechSolutions S.L.',
            recipient: 'Global Retail Corp.',
            total: '14,850.00'
          }
        })
        setStep('result')
      }, 2000)
    }
  }, [step])

  return (
    <div>
      {step === 'upload' && (
        <div>
          <h1>Upload a document</h1>
          <button onClick={() => setStep('processing')}>Upload</button>
        </div>
      )}

      {step === 'processing' && (
        <div>
          <h1>Classifying your document...</h1>
        </div>
      )}

      {step === 'result' && result && (
        <div>
          <h1>Category: {result.category}</h1>
          <h2>Confidence: {result.confidence}%</h2>

          {result.category === 'invoice' && (
            <div>
              <h3>Extracted fields</h3>
              {Object.entries(result.fields).map(([key, value]) => (
                <p key={key}>
                  <strong>{FIELD_LABELS[key]}:</strong> {value ?? 'Not found'}
                </p>
              ))}
            </div>
          )}

          <button onClick={() => setStep('upload')}>Start over</button>
        </div>
      )}
    </div>
  )
}