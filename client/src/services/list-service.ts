export type List = {
  id: number;
  identificacion: string;
  nombre: string;
};

class ListService {
  async getList(): Promise<List[]> {
    return (await fetch("http://127.0.0.1:8000/api/v1/identificaciones")).json();
  }
  async searchList(search: string): Promise<List[]> {
    return (await fetch(`http://127.0.0.1:8000/api/v1/identificaciones?search=${search}`)).json();
  }
}

export default new ListService();
