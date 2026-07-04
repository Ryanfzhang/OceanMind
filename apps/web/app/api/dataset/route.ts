import { proxyGet } from "@/lib/backend-proxy";

export async function GET() {
  return proxyGet("/dataset", "Backend dataset request failed");
}
