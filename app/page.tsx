"use client";

import { useState } from "react";

export default function Home() {
  const [pre, setPre] = useState("");
  const [program, setProgram] = useState("");
  const [post, setPost] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [verified, setVerified] = useState(false);
  const [invariants, setInvariants] = useState<string>(""); // new state for invariants

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setInvariants(""); // clear old invariants

    const res = await fetch("/api/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pre, program, post }),
    });

    const data = await res.json();

    if (data.data["success"]) {
      setResult("Program verified");
      setVerified(true);
    } else {
      setResult("Could not verify program");
      setVerified(false);
    }

    // Set invariants if returned from backend
    if (data.data["invariants"]) {
      setInvariants(data.data["invariants"]);
    }

    setLoading(false);
  }

  return (
    <main className="min-h-screen bg-gray-100 flex items-center justify-center p-10">
      <div className="w-full max-w-5xl bg-white shadow-xl rounded-2xl p-8 flex gap-8">
        {/* Left side: form */}
        <div className="flex-1">
          <h1 className="text-3xl font-bold mb-6 text-center">Program Verifier</h1>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block font-semibold mb-1">Precondition</label>
              <textarea
                value={pre}
                onChange={(e) => setPre(e.target.value)}
                className="w-full border rounded-lg p-3 focus:ring-2 focus:ring-blue-500"
                rows={2}
              />
            </div>

            <div>
              <label className="block font-semibold mb-1">Program</label>
              <textarea
                value={program}
                onChange={(e) => setProgram(e.target.value)}
                className="w-full border rounded-lg p-3 focus:ring-2 focus:ring-blue-500"
                rows={4}
              />
            </div>

            <div>
              <label className="block font-semibold mb-1">Postcondition</label>
              <textarea
                value={post}
                onChange={(e) => setPost(e.target.value)}
                className="w-full border rounded-lg p-3 focus:ring-2 focus:ring-blue-500"
                rows={2}
              />
            </div>

            <button
              type="submit"
              className="w-full bg-blue-600 text-white py-3 rounded-lg text-lg font-semibold
                         hover:bg-blue-700 active:bg-blue-800 active:scale-95 transition transform"
              disabled={loading}
            >
              {loading ? "Checking..." : "Verify"}
            </button>
          </form>

          {result && (
            <div
              className={`mt-6 p-4 rounded-xl text-center text-white text-lg font-semibold ${
                verified ? "bg-gradient-to-r from-green-500 to-emerald-500" : "bg-gradient-to-r from-red-500 to-red-500"
              }`}
            >
              Result: {result}
            </div>
          )}
        </div>

        {/* Right side: invariants sidebar */}
        <div className="w-64 bg-gray-50 p-4 rounded-xl border border-gray-200 shadow-inner">
          <h2 className="text-xl font-semibold mb-4 text-center">Invariants</h2>
          {invariants !== "" ? (
            <pre className="text-sm text-blue-800 whitespace-pre-wrap">{invariants}</pre>
          ) : (
            <p className="text-gray-500 text-center">No invariants yet</p>
          )}
        </div>
      </div>
    </main>
  );
}
