// The typed API client. Paths, parameters and response shapes all come from
// schema.ts, generated from the FastAPI app (`npm run gen:api`), so a renamed
// field on the Python side is a compile error here rather than an `undefined`.
import createClient, { type Middleware } from "openapi-fetch";

import { idToken } from "../auth";
import type { components, paths } from "./schema";

export type Schemas = components["schemas"];

const withToken: Middleware = {
  async onRequest({ request }) {
    const token = await idToken();
    if (token) request.headers.set("Authorization", `Bearer ${token}`);
    return request;
  },
};

export const api = createClient<paths>({ baseUrl: "" });
api.use(withToken);

/** A non-2xx answer, carrying the API's own `detail` message. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

function detailOf(body: unknown, status: number): string {
  const detail = (body as { detail?: unknown } | undefined)?.detail;
  if (typeof detail === "string") return detail;
  // FastAPI's validation errors: a list of { loc, msg }.
  if (Array.isArray(detail)) return detail.map((d: { msg?: string }) => d.msg).join("; ");
  return `The API answered ${status}.`;
}

/** The data of an openapi-fetch result, or an ApiError for the caller to show. */
export async function unwrap<T>(
  call: Promise<{ data?: T; error?: unknown; response: Response }>,
): Promise<T> {
  const { data, error, response } = await call;
  if (error !== undefined || data === undefined) {
    throw new ApiError(response.status, detailOf(error, response.status));
  }
  return data;
}
