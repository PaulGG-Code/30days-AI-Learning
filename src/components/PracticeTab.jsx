import { useState, useRef } from "react";

function openInColab() {
  const baseUrl = "https://colab.research.google.com/#create=true";
  window.open(baseUrl, '_blank');
}

export default function PracticeTab({ initialCode }) {
  const [code, setCode] = useState(initialCode || "print('Hello, world!')");
  const [output, setOutput] = useState("");
  const [copied, setCopied] = useState(false);
  const pyodideRef = useRef(null);

  async function ensurePyodide() {
    if (!window.loadPyodide) {
      await new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = "https://cdn.jsdelivr.net/pyodide/v0.25.1/full/pyodide.js";
        script.onload = resolve;
        script.onerror = reject;
        document.body.appendChild(script);
      });
    }
    if (!pyodideRef.current) {
      setOutput("Loading Python runtime...");
      pyodideRef.current = await window.loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.25.1/full/",
      });
    }
    return pyodideRef.current;
  }

  async function runCode() {
    const pyodide = await ensurePyodide();
    const packagesToCheck = [
      { name: 'scikit-learn', match: /sklearn/ },
      { name: 'numpy', match: /numpy|np\./ },
      { name: 'pandas', match: /pandas|pd\./ },
    ];
    for (const pkg of packagesToCheck) {
      if (pkg.match.test(code)) {
        setOutput(`Loading ${pkg.name}...`);
        await pyodide.loadPackage(pkg.name);
      }
    }
    try {
      let output = "";
      pyodide.setStdout({ batched: (s) => { output += s; } });
      pyodide.setStderr({ batched: (s) => { output += s; } });
      pyodide.runPython(code);
      let processed = output
        .replace(/(Warning:)/g, '\n$1')
        .replace(/(ConvergenceWarning:)/g, '\n$1')
        .replace(/(Accuracy:)/g, '\n$1')
        .replace(/(Precision:)/g, '\n$1')
        .replace(/(Recall:)/g, '\n$1')
        .replace(/(F1-score:)/g, '\n$1');
      setOutput(processed.trim() || "(no output)");
    } catch (e) {
      setOutput(e.toString());
    } finally {
      pyodide.setStdout();
      pyodide.setStderr();
    }
  }

  function copyToClipboard() {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  return (
    <div>
      <textarea
        value={code}
        onChange={e => setCode(e.target.value)}
        rows={10}
        className="w-full font-mono bg-gray-100 p-2 rounded"
      />
      <div className="flex gap-2 mt-2 items-center">
        <button
          onClick={runCode}
          className="px-4 py-2 bg-blue-600 text-white rounded"
        >
          Run
        </button>
        <button
          onClick={copyToClipboard}
          className="px-4 py-2 bg-gray-200 text-gray-800 rounded border border-gray-300"
          title="Copy code to clipboard"
        >
          {copied ? "Copied!" : "Copy to Clipboard"}
        </button>
        <button
          onClick={() => openInColab()}
          className="px-4 py-2 bg-yellow-500 text-black rounded"
          title="Open this code in Google Colab for full Python support (including TensorFlow, PyTorch, etc.)"
        >
          Run in Colab
        </button>
      </div>
      <pre className="bg-gray-900 text-green-400 p-4 rounded mt-4 min-h-[2em]" style={{ whiteSpace: 'pre-wrap' }}>
        {output}
      </pre>
    </div>
  );
} 