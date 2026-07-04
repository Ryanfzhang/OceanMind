import { proxyGet } from "@/lib/backend-proxy";

export async function GET() {
  return proxyGet("/health", "Backend health check failed");
}
