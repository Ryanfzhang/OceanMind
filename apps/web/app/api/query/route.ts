import { NextRequest } from "next/server";
import { proxyJsonPost } from "@/lib/backend-proxy";

export async function POST(request: NextRequest) {
  return proxyJsonPost(request, "/query", "Backend query failed");
}
