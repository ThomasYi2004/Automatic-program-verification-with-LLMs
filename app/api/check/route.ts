import { NextResponse } from "next/server";

export async function POST(req: Request) {
    const body = await req.json();
    console.log(body)
    const response = await fetch("http://localhost:8000/check", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });
  const data = await response.json()
  console.log("response: ", data)
  
  return NextResponse.json({ data });
}
