"use client";

import { useState, useEffect } from "react";

const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

type ContextSource = {
  source_id: string;
  chunk: {
    text: string;
    metadata: {
      source: string;
      page_number: number;
    };
    chunk_number: number;
  }
}

type EvaluationStatus =
  | "pending"
  | "completed"
  | "failed"
  | "skipped";

type EvaluationScores = {
  faithfulness: number;
  answer_relevancy: number;
  context_precision: number;
}

type EvaluationState = {
  evaluation_id: string;
  status: EvaluationStatus;
  scores: EvaluationScores | null;
  reason: string | null;
}

type DocumentInfo = {
  document_id: string;
  filename: string | null;
  page_count: number;
  status: "uploaded" | "ready";
}

export default function HomePage() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [sources, setSources] = useState<ContextSource[]>([]);
  const [evaluation, setEvaluation] = useState<EvaluationState | null>(null);
  const [pollingError, setPollingError] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [activeDocument, setActiveDocument] = useState<DocumentInfo | null>(null);
  const [documentPhase, setDocumentPhase] = useState<"idle" | "uploading" | "indexing">("idle");
  const [uploadError, setUploadError] = useState("");

  const isPreparing = documentPhase !== "idle";

  const evaluationId = evaluation?.evaluation_id;
  const evaluationStatus = evaluation?.status;


  useEffect(() => {
    if(!evaluationId || evaluationStatus !== "pending") return;

    let cancelled = false;
    let timer: number | undefined;
    let checks = 0;

    const controller = new AbortController();

    async function poll() {
      if (cancelled) return;

      if (checks >= 60) {
        setPollingError(
          "Stopped checking after 60 attempts. Evaluation may still be running."
        );
        return;
      }

      checks += 1;

      try {
        const response = await fetch(
          `${API_URL}/evaluations/${evaluationId}`,
          { signal: controller.signal }
        );
        if(!response.ok) {
            throw new Error(
            `Could not fetch evaluation status (${response.status}).`
          );
        }

        const latest: EvaluationState = await response.json();
        if(cancelled) return;

        if (latest.evaluation_id !== evaluationId) {
          throw new Error("Received a different evaluation ID.");
        }

        setEvaluation((current) =>
          current?.evaluation_id === evaluationId ? latest : current
        );

        if (latest.status === "pending") {
          timer = window.setTimeout(poll, 5000);
        }
      } catch (caughtError) {
        if (cancelled) return;
        setPollingError(
          caughtError instanceof Error
            ? caughtError.message
            : "Could not refresh evaluation status."
        );
      }
    }

    timer = window.setTimeout(poll, 2000);
    return () => {
      cancelled=true;

      window.clearTimeout(timer);
      controller.abort();
    }
  }, [evaluationId, evaluationStatus])

  function clearAnswer() {
    setAnswer("");
    setSources([]);
    setEvaluation(null);
    setPollingError("");
    setError("");
  }

  async function handlePrepareDocument() {
    if (!selectedFile || isPreparing || isLoading) return;

    if (selectedFile.size > 10 * 1024 * 1024) {
      setUploadError("Choose a PDF no larger than 10 MiB.");
      return;
    }

    clearAnswer();
    setActiveDocument(null);
    setUploadError("");
    setDocumentPhase("uploading");

    try {
      const formData = new FormData();

      formData.append("file", selectedFile)

      const uploadResponse = await fetch(
        `${API_URL}/documents`,
        {
          method: "POST",
          body: formData
        }
      )

      if(!uploadResponse.ok) {
        throw new Error(`Upload failed (${uploadResponse.status}).`);
      }

      const uploaded: DocumentInfo = await uploadResponse.json();
      
      if (
        typeof uploaded?.document_id !== "string" ||
        uploaded.status !== "uploaded"
      ) {
        throw new Error("Unexpected upload response.")
      }

      setDocumentPhase("indexing");

      const ingestResponse = await fetch(
        `${API_URL}/documents/${uploaded.document_id}/ingest`,
        {
          method: "POST"
        }
      )

      if (!ingestResponse.ok) {
        throw new Error(
          `Document preparation failed (${ingestResponse.status}).`
        )
      }

      const ready: DocumentInfo = await ingestResponse.json();

      if (
        ready?.document_id !== uploaded.document_id ||
        ready.status !== "ready"
      ) {
        throw new Error("The document is not ready for questions.");
      }
      setActiveDocument(ready);
    } catch (caughtError) {
      setUploadError(
        caughtError instanceof Error
        ? caughtError.message
        : "Could not prepare the document."
      )
    } finally {
      setDocumentPhase("idle")
    }
  }

  async function handleAsk() {
    const submittedQuestion = question.trim();
    
    if (
      !submittedQuestion || 
      isLoading ||
      isPreparing ||
      activeDocument?.status !== "ready"
    ) return;

    clearAnswer()
    setIsLoading(true);

    try {
      const response = await fetch(`${API_URL}/questions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: submittedQuestion,
          document_id: activeDocument.document_id,
        }),
      });

      if (!response.ok) {
        throw new Error(`Request failed (${response.status}).`);
      }

      const data = await response.json();

      if (
        typeof data?.answer !== "string" ||
        !Array.isArray(data?.sources)
      ) {
        throw new Error("The server returned an unexpected response.");
      }

      setAnswer(data.answer);
      setSources(data.sources)
      setEvaluation({
        evaluation_id: data.evaluation_id,
        status: data.evaluation_status,
        scores: data.scores,
        reason: null
      });
    } catch (caughtError) {
      setAnswer("");
      setError(
        caughtError instanceof Error
          ? caughtError.message
          : "Could not get an answer.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="policy-page">
      <h1>HR Policy Bot</h1>
      <p>Ask a question about the employee handbook.</p>

      <section className="panel">
        <h2>Your document</h2>

        <label htmlFor="pdf-upload">Choose an HR-policy PDF</label>

        <input
          id="pdf-upload"
          type="file"
          accept=".pdf,application/pdf"
          disabled={isPreparing || isLoading}
          onChange={(event) => {
            setSelectedFile(event.target.files?.[0] ?? null);
            setActiveDocument(null);
            setUploadError("");
            clearAnswer();
          }}
        />

        <button
          type="button"
          onClick={handlePrepareDocument}
          disabled={!selectedFile || isPreparing || isLoading}
        >
          {documentPhase === "uploading"
          ? "Uploading..."
          : documentPhase === "indexing"
            ? "Preparing document..."
            : "Upload and prepare"}
        </button>

        {uploadError && <p role="alert">{uploadError}</p>}

        {
          activeDocument && (
            <p role="status">
              Ready: {activeDocument.filename ?? "PDF document"}
              {" - "}
              {activeDocument.page_count} pages
            </p>
          )
        }
      </section>

      <div>
        <label htmlFor="question">Your Question</label>
      </div>

      <textarea
        id="question"
        rows={5}
        maxLength={2000}
        placeholder="How can I review my personal record?"
        value={question}
        onChange={(event) => {
          setQuestion(event.target.value);
        }}
      />
      <p>Characters: {question.length} / 2000</p>

      <button
        type="button"
        onClick={handleAsk}
        disabled={
          question.trim() === "" || 
          isLoading ||
          isPreparing ||
          activeDocument?.status !== "ready"
        }
      >
        {isLoading ? "Finding an answer..." : "Ask"}
      </button>

      {error && <p role="alert">{error}</p>}

      {answer && (
        <section className="panel">
          <h2>Answer</h2>
          <p style={{whiteSpace: "pre-wrap"}}>{answer}</p>

          <h3>Sources</h3>
          {sources.length === 0 ? (
            <p>No sources cited.</p>
          ): (
            sources.map((source) => (
              <details key={source.source_id}>
                <summary>
                  {`[${source.source_id}] ${source.chunk.metadata.source} — PDF page ${source.chunk.metadata.page_number}`}
                </summary>
                <p style={{whiteSpace: "pre-wrap"}}>
                  {source.chunk.text}
                </p>
              </details>
            ))
          )}

          {evaluation && (
            <section aria-live="polite">
              <h3>Answer evaluation</h3>

              {evaluation.status === "pending" && (
                <p>Evaluating the answer...</p>
              )}

              {evaluation.status === "completed" && evaluation.scores && (
                <dl>
                  <dt>Faithfulness</dt>
                  <dd>
                    {evaluation.scores.faithfulness.toFixed(3)}
                  </dd>
                  
                  <dt>Answer Relevancy</dt>
                  <dd>
                    {evaluation.scores.answer_relevancy.toFixed(3)}
                  </dd>

                  <dt>Context Precision</dt>
                  <dd>
                    {evaluation.scores.context_precision.toFixed(3)}
                  </dd>
                </dl>
              )}

              {(evaluation.status === "skipped" ||
                evaluation.status === "failed") && (
                <p>
                  Evaluation {evaluation.status}: {" "}
                  {evaluation.reason ?? "No further details available."}
                </p>
              )}
              {pollingError && <p role="alert">{pollingError}</p>}
            </section>
          )}
        </section>
      )}
    </main>
  );
}
