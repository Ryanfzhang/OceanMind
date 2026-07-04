import { proxyBinaryPost } from "@/lib/backend-proxy";

export async function POST(request: Request) {
  return proxyBinaryPost(request, "/report/export", "Backend report export failed");
}
