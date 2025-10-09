import { NavLink } from "react-router-dom";

const Home = () => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-slate-900 text-white px-4">
      <h1 className="text-3xl sm:text-5xl font-bold mb-12 text-center">
        Choose Your Chat
      </h1>

      <div className="flex flex-col sm:flex-row gap-6 w-full max-w-md">
        <NavLink
          to="/graphicalchat"
          className="flex-1 bg-blue-700 rounded-2xl shadow-lg hover:bg-blue-600 active:scale-95 transition-all duration-200 flex flex-col items-center justify-center py-10"
        >
          <span className="text-7xl mb-3">💬</span>
          <span className="text-xl font-semibold">Graphical Chat</span>
        </NavLink>

        <NavLink
          to="/terminalchat"
          className="flex-1 bg-green-700 rounded-2xl shadow-lg hover:bg-green-600 active:scale-95 transition-all duration-200 flex flex-col items-center justify-center py-10"
        >
          <span className="text-7xl mb-3">💻</span>
          <span className="text-xl font-semibold">Terminal Chat</span>
        </NavLink>
      </div>
    </div>
  );
};

export default Home;
