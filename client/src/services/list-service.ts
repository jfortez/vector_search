import { API_URL } from "@/config";

export type List = {
  id: number;
  identificacion: string;
  nombre: string;
};

type MutationResult = {
  message: string;
};

const ROUTE = "identificaciones";
const SERVICE_URL = `${API_URL}/${ROUTE}`;

export type Modes = "fuzzy" | "faiss";
export type SearchParams = {
  search?: string;
  mode?: Modes;
  threshold?: number;
};
class ListService {
  async getList(): Promise<List[]> {
    const res = await fetch(SERVICE_URL);
    if (!res.ok) {
      throw new Error(res.statusText);
    }
    return await res.json();
  }

  async searchList({
    search = "",
    mode = "faiss",
    threshold = 0.1,
  }: SearchParams): Promise<List[]> {
    const url = new URL(SERVICE_URL);
    url.searchParams.set("search", search);
    url.searchParams.set("mode", mode);
    url.searchParams.set("threshold", threshold.toString());
    return (await fetch(url.toString())).json();
  }
  async insertItem(item: Omit<List, "id">): Promise<MutationResult> {
    const result = await fetch(SERVICE_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(item),
    });
    if (!result.ok) {
      throw new Error(result.statusText);
    }
    return result.json();
  }

  async updateItem(item: List): Promise<MutationResult> {
    const result = await fetch(`${SERVICE_URL}/${item.id}`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(item),
    });
    if (!result.ok) {
      throw new Error(result.statusText);
    }
    return result.json();
  }

  async deleteItem(id: number): Promise<MutationResult> {
    const result = await fetch(`${SERVICE_URL}/${id}`, {
      method: "DELETE",
    });
    if (!result.ok) {
      throw new Error(result.statusText);
    }
    return result.json();
  }
}

export default new ListService();
