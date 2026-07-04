import { NextRequest } from "next/server";
import { proxyStreamPost } from "@/lib/backend-proxy";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  return proxyStreamPost(request, "/query/stream", "Backend query stream failed");
}
