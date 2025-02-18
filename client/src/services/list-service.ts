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

class ListService {
  async getList(): Promise<List[]> {
    return (await fetch(SERVICE_URL)).json();
  }
  async searchList(search?: string): Promise<List[]> {
    return (await fetch(`${SERVICE_URL}?search=${search}`)).json();
  }
  async insertItem(item: List): Promise<MutationResult> {
    const result = await fetch(SERVICE_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(item),
    });
    return result.json();
  }

  async updateItem(item: List): Promise<MutationResult> {
    const result = await fetch(SERVICE_URL, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(item),
    });
    return result.json();
  }

  async deleteItem(id: number): Promise<MutationResult> {
    const result = await fetch(`${SERVICE_URL}/${id}`, {
      method: "DELETE",
    });
    return result.json();
  }
}

export default new ListService();
